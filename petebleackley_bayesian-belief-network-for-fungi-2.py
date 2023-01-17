%matplotlib inline
import pandas
import csv
import collections
import pprint
import numpy
class MushroomOrToadstool(object):
    """Classifier to determine edibility of fungi"""
    def __init__(self,filename):
        self.hidden_variables=[]
        data=pandas.read_csv(filename,index_col=False)
        variables=data.columns[1:]
        links={}
        for column in variables:
            link={'neighbour':None,
                  'score':0}
            for neighbour in links:
                score=self.MutualInformation(data,[column,neighbour])
                if score>link['score']:
                    link['neighbour']=neighbour
                    link['score']=score
                if score>links[neighbour]['score']:
                    links[neighbour]['neighbour']=column
                    links[neighbour]['score']=score
            links[column]=link
        column_clusters=[]
        for (column,link) in links.items():
            if link['neighbour'] is not None:
                cclust=None
                nclust=None
                for cluster in column_clusters:
                    if column in cluster:
                        cclust=cluster
                    if link['neighbour'] in cluster:
                        nclust=cluster
                if cclust is None and nclust is None:
                    column_clusters.append([column,link['neighbour']])
                elif cclust is None:
                    nclust.append(column)
                elif nclust is None:
                    cclust.append(link['neighbour'])
                elif cclust!=nclust:
                    cclust.extend(nclust)
                    column_clusters.remove(nclust)
        print(len(column_clusters),'hidden variables inferred')
        pprint.pprint(column_clusters)
        self.column_to_hidden_variable={}
        for (i,cluster) in enumerate(column_clusters):
            for column in cluster:
                self.column_to_hidden_variable[column]=i
        for cluster in column_clusters:
            cluster.append('class')
        clusters=[[] for cluster in column_clusters]
        for (i,row) in data.iterrows():
            for (j,column_cluster) in enumerate(column_clusters):
                best=-1
                sim=0.5
                for (k,cluster) in enumerate(clusters[j]):
                    x=sum(cluster[key][value]/sum(cluster[key].values())
                          for (key,value) in row[column_cluster].iteritems())/len(column_cluster)
                    if x>sim:
                        best=k
                        sim=x
                if best==-1:
                    clusters[j].append(collections.defaultdict(lambda: collections.defaultdict(float)))
                    print(i+1,'rows analysed',len(clusters[j]),'clusters found for hidden variable',j)
                for (key,value) in row[column_cluster].iteritems():
                    clusters[j][best][key][value]+=1.0
        for (i,cluster) in enumerate(clusters):
            index=[]
            for column in column_clusters[i]:
                index.extend([(column,value) for value in data[column].unique()])
            self.hidden_variables.append(pandas.DataFrame({(key,value):[group[key][value]+1.0 
                                                                        for group in cluster]
                                                           for (key,value) in index}).T)
        self.priors=[hv.sum(axis=0) for hv in self.hidden_variables]
        for (i,prior) in enumerate(self.priors):
            self.priors[i]/=prior.sum()
        self.edibility_prior=data['class'].value_counts(normalize=1)
        def normalize(group):
            return group.div(group.sum(axis=0),axis='columns')
        for (i,hv) in enumerate(self.hidden_variables):
            self.hidden_variables[i]=hv.groupby(axis=0,level=0).apply(normalize)
            
    def MutualInformation(self,data,columns):
        individual={column:data[column].value_counts(normalize=True)
                   for column in columns}        
        [first,second]=columns
        joint=data.groupby(columns)[first].count().unstack(fill_value=0.0)/data.shape[0]
        joint.sort_index(inplace=True)
        joint.sort_index(axis=1,inplace=True)
        product=pandas.DataFrame({key:value*individual[second]
                                 for (key,value) in individual[first].iteritems()}).T
        ratio=joint/product
        return (joint*pandas.DataFrame(numpy.log2(ratio.values),
                                       index=ratio.index,
                                       columns=ratio.columns)).sum().sum()
    
    def __call__(self,**kwargs):
        "Estimates the probability that a fungus is edible given the features in kwargs"
        categories=[prior.copy() for prior in self.priors]
        for (key,value) in kwargs.items():
            if key in self.column_to_hidden_variable:
                hv=self.column_to_hidden_variable[key]
                categories[hv]*=self.hidden_variables[hv].loc[(key,value)]
                categories[hv]/=categories[hv].sum()
        result=self.edibility_prior.copy()
        for (probabilities,category) in zip(self.hidden_variables,categories):
            result*=(probabilities.loc['class']*category).sum(axis=1)
            result/=result.sum()
        return result
    
    def test(self,filename):
        """Produces KDE plots of the estimated probability"""
        data=pandas.read_csv(filename,index_col=False)
        observables=[column for column in data.columns if column!='class']
        results=pandas.DataFrame([self(**row) for (i,row) in data[observables].iterrows()],
                                 index=range(data.shape[0]))
        results.loc[:,'class']=data['class']
        return results
        
            
BBN=MushroomOrToadstool('../input/mushrooms.csv')
BBN.edibility_prior.plot.bar()
BBN.priors[0].plot.bar()
BBN.hidden_variables[0].loc['class'].T.plot.bar()
BBN.priors[1].plot.bar()
BBN.hidden_variables[1].loc['class'].T.plot.bar()
result=BBN.test('../input/mushrooms.csv')
result['e'].plot.kde()
result[result['class']=='e']['e'].plot.kde()
result[result['class']=='p']['e'].plot.kde()
result[result['e']>0.5]['class'].value_counts(normalize=True).plot.bar()
result[result['e']>0.9]['class'].value_counts(normalize=True).plot.bar()