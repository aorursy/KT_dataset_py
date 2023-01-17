# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
%matplotlib inline

import numpy
import numpy.linalg
import pandas
import scipy.cluster.hierarchy
import sklearn
import sklearn.cluster
import sklearn.linear_model
import re
import matplotlib.pyplot
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

def ReduceDimensions(data,keep_variance=0.8,tolerance=1.0E-12,max_iter=1024):
    """Reduces dimensions of data"""
    residuals=data-data.mean(axis=1,keepdims=True)
    noise=(1.0-keep_variance)*residuals.var()
    convergence=1.0-tolerance
    result=None
    n=0
    while n<max_iter and residuals.var()>noise:
        component=residuals.var(axis=1,keepdims=True)
        if result is not None:
            component-=(result*(component.T.dot(result))).sum(axis=1,keepdims=True)
        component/=numpy.linalg.norm(component)
        corr=0.0
        n=0
        while n<max_iter and corr<convergence:
            projection=residuals.dot(component.T.dot(residuals).T)
            projection/=numpy.linalg.norm(projection)
            corr=projection.T.dot(component)
            component=projection
            n+=1
        if n<max_iter:
            if result is None:
                result=component
            else:
                result=numpy.hstack([result,component])
            residuals-=numpy.outer(component,component.T.dot(residuals))
    return result
split_name=re.compile('[.-]')
   
def rename_patients(name):
    components=split_name.split(name)
    return '-'.join(('TCGA',components[0],components[1]))
    
class BreastCancerAnalysis(object):
    """Analyses the breast cancer proteome date from Kaggle"""
    def __init__(self):
        """Loads the data tables"""
        self.protein_activity=pandas.read_csv("../input/77_cancer_proteomes_CPTAC_itraq.csv")
        self.protein_activity.set_index(['RefSeq_accession_number','gene_symbol','gene_name'],inplace=True)
        self.protein_activity.fillna(self.protein_activity.median(),inplace=True)
        self.protein_activity.rename(columns=rename_patients,inplace=True)
        self.clinical_data=pandas.read_csv("../input/clinical_data_breast_cancer.csv").set_index('Complete TCGA ID')
        self.principal_components=None
        self.protein_clusters=None
        self.patient_protein_activity=None
        self.clinical_prediction_models=None
        
    def fit_principal_components(self):
        """Reduces the dimensionality of the data"""
        self.principal_components=pandas.DataFrame(ReduceDimensions(self.protein_activity.values),
                                                   index=self.protein_activity.index)
        return self.principal_components
        
    def dendrogram(self):
        """Performs heirarchical clustering and plots a dendrogram to chose number
           of clusters to fit"""
        result=matplotlib.pyplot.gca()
        scipy.cluster.hierarchy.dendrogram(scipy.cluster.hierarchy.ward(self.principal_components),ax=result)
        return result
        
    def cluster_proteins(self,n=2):
        """Finds clusters of proteins whose activity is related"""
        clusters=sklearn.cluster.AgglomerativeClustering(n_clusters=n,
                                                         memory='/tmp')
        self.protein_clusters=pandas.DataFrame(clusters.fit_predict(self.principal_components),
                                               index=self.principal_components.index,
                                               columns=['cluster'])
        return self.protein_clusters
        
    
    def patient_cluster_activity(self):
        """Calculates the activity of each protein cluster for each patient"""
        mean_subtracted_protein_activity=self.protein_activity.sub(self.protein_activity.values.mean(axis=1),
                                                                   axis='index')
        patient_components=mean_subtracted_protein_activity.T.dot(self.principal_components)
        kernel=self.principal_components.merge(self.protein_clusters,
                                               left_index=True,
                                               right_index=True).groupby('cluster').mean().T
        self.patient_protein_activity=patient_components.dot(kernel)
        return self.patient_protein_activity

    def train_clinical_models(self):
        """Fits a model to predict each clinical feature from the protein activity"""
        self.clinical_prediction_models={'Gender':sklearn.linear_model.LogisticRegression(solver='lbfgs',multi_class='multinomial'),
                                         'Age at Initial Pathologic Diagnosis':sklearn.linear_model.LassoLars(copy_X=True),
                                         'ER Status':sklearn.linear_model.LogisticRegression(solver='lbfgs',multi_class='multinomial'),
                                         'PR Status':sklearn.linear_model.LogisticRegression(solver='lbfgs'),
                                         'HER2 Final Status':sklearn.linear_model.LogisticRegression(solver='lbfgs'),
                                         'Tumor':sklearn.linear_model.LogisticRegression(solver='lbfgs',multi_class='multinomial'),
                                         'Node':sklearn.linear_model.LogisticRegression(solver='lbfgs',multi_class='multinomial'),
                                         'Metastasis-Coded':sklearn.linear_model.LogisticRegression(solver='lbfgs',multi_class='multinomial'),
                                         'AJCC Stage':sklearn.linear_model.LogisticRegression(solver='lbfgs',multi_class='multinomial'),
                                         'Vital Status':sklearn.linear_model.LogisticRegression(solver='lbfgs',multi_class='multinomial'),
                                         'OS Time':sklearn.linear_model.LassoLars(copy_X=True),
                                         'PAM50 mRNA':sklearn.linear_model.LogisticRegression(solver='lbfgs',multi_class='multinomial'),
                                         'SigClust Unsupervised mRNA':sklearn.linear_model.LassoLars(copy_X=True),
                                         'miRNA Clusters':sklearn.linear_model.LassoLars(copy_X=True),
                                         'methylation Clusters':sklearn.linear_model.LassoLars(copy_X=True),
                                         'RPPA Clusters':sklearn.linear_model.LogisticRegression(solver='lbfgs',multi_class='multinomial'),
                                         'Integrated Clusters (with PAM50)':sklearn.linear_model.LassoLars(copy_X=True),
                                         'Integrated Clusters (no exp)':sklearn.linear_model.LassoLars(copy_X=True),
                                         'Integrated Clusters (unsup exp)':sklearn.linear_model.LassoLars(copy_X=True)}
        combined_data=self.patient_protein_activity.merge(self.clinical_data,
                                                          how='inner',
                                                          left_index=True,
                                                          right_index=True)
        for (column,model) in self.clinical_prediction_models.items():
            model.fit(combined_data[self.patient_protein_activity.columns].values,
                      combined_data[column].values)
        return pandas.Series({column:model.score(combined_data[self.patient_protein_activity.columns].values,
                                                 combined_data[column].values)
                              for (column,model) in self.clinical_prediction_models.items()}).plot.barh()
    
    def visualize(self,model):
        """Plots a chart of the chosen model"""
        data=None
        coefficients=self.clinical_prediction_models[model]
        if coefficients.__class__.__name__=='LogisticRegression':
            data=pandas.DataFrame(coefficients.coef_.T,
                                  columns=coefficients.classes_[:coefficients.coef_.shape[0]],
                                  index=self.patient_protein_activity.columns)
        else:
            data=pandas.Series(coefficients.coef_,
                               index=self.patient_protein_activity.columns)
        return data.plot.bar(figsize=(6,4))
models=BreastCancerAnalysis()
models.fit_principal_components()
models.dendrogram()
models.cluster_proteins(8)
models.patient_cluster_activity()
models.train_clinical_models()
models.visualize('Gender')
models.visualize('ER Status')
models.visualize('PR Status')
models.visualize('HER2 Final Status')
models.visualize('Tumor')
models.visualize('Node')
models.visualize('Metastasis-Coded')
models.visualize('AJCC Stage')
models.visualize('Vital Status')
models.visualize('OS Time')
models.visualize('PAM50 mRNA')
models.visualize('RPPA Clusters')