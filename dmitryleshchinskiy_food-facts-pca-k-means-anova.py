# Library of Functions for the OpenClassrooms Multivariate Exploratory Data Analysis Course



import matplotlib.pyplot as plt

from matplotlib.collections import LineCollection

import numpy as np

import pandas as pd

from scipy.cluster.hierarchy import dendrogram

from pandas.plotting import parallel_coordinates

import seaborn as sns





palette = sns.color_palette("bright", 10)



def display_circles(pcs, n_comp, pca, axis_ranks, labels=None, label_rotation=0, lims=None):

    """Display correlation circles, one for each factorial plane"""



    # For each factorial plane

    for d1, d2 in axis_ranks: 

        if d2 < n_comp:



            # Initialise the matplotlib figure

            fig, ax = plt.subplots(figsize=(20,20))



            # Determine the limits of the chart

            if lims is not None :

                xmin, xmax, ymin, ymax = lims

            elif pcs.shape[1] < 30 :

                xmin, xmax, ymin, ymax = -1, 1, -1, 1

            else :

                xmin, xmax, ymin, ymax = min(pcs[d1,:]), max(pcs[d1,:]), min(pcs[d2,:]), max(pcs[d2,:])



            # Add arrows

            # If there are more than 30 arrows, we do not display the triangle at the end

            if pcs.shape[1] < 30 :

                plt.quiver(np.zeros(pcs.shape[1]), np.zeros(pcs.shape[1]),

                   pcs[d1,:], pcs[d2,:], 

                   angles='xy', scale_units='xy', scale=1, color="grey")

                # (see the doc : https://matplotlib.org/api/_as_gen/matplotlib.pyplot.quiver.html)

            else:

                lines = [[[0,0],[x,y]] for x,y in pcs[[d1,d2]].T]

                ax.add_collection(LineCollection(lines, axes=ax, alpha=.1, color='black'))

            

            # Display variable names

            if labels is not None:  

                for i,(x, y) in enumerate(pcs[[d1,d2]].T):

                    if x >= xmin and x <= xmax and y >= ymin and y <= ymax :

                        plt.text(x, y, labels[i], fontsize='14', ha='center', va='center', rotation=label_rotation, color="blue", alpha=0.5)

            

            # Display circle

            circle = plt.Circle((0,0), 1, facecolor='none', edgecolor='b')

            plt.gca().add_artist(circle)



            # Define the limits of the chart

            plt.xlim(xmin, xmax)

            plt.ylim(ymin, ymax)

        

            # Display grid lines

            plt.plot([-1, 1], [0, 0], color='grey', ls='--')

            plt.plot([0, 0], [-1, 1], color='grey', ls='--')



            # Label the axes, with the percentage of variance explained

            plt.xlabel('PC{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))

            plt.ylabel('PC{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            nr=d1+1

            plt.title("Correlation Circle (PC{} and PC{})".format(d1+1, d2+1))

            plt.show(block=False)

            d = {'values': pca.components_[d1], 'factors': labels}

            df1= pd.DataFrame(d)

            df1.set_index('factors')

            df2=df1.sort_values(by='values', ascending=False)

            df3=df1.sort_values(by='values', ascending=True)

            print("Principal Component" + str(nr)+ " Presenting Values")

            print(df2.head(3))

            print(df3.head(3))

            

            nr=d2+1

            

            d = {'values': pca.components_[d2], 'factors': labels}

            df1= pd.DataFrame(d)

            df1.set_index('factors')

            df2=df1.sort_values(by='values', ascending=False)

            df3=df1.sort_values(by='values', ascending=True)

            print("Principal Component" + str(nr)+ " Presenting Values")

            print(df2.head(3))

            print(df3.head(3))

        

def display_factorial_planes(X_projected, n_comp, pca, axis_ranks, labels=None, alpha=1, illustrative_var=None):

    '''Display a scatter plot on a factorial plane, one for each factorial plane'''



    # For each factorial plane

    for d1,d2 in axis_ranks:

        if d2 < n_comp:

 

            # Initialise the matplotlib figure      

            fig = plt.figure(figsize=(7,6))

        

            # Display the points

            if illustrative_var is None:

                plt.scatter(X_projected[:, d1], X_projected[:, d2], alpha=alpha)

            else:

                illustrative_var = np.array(illustrative_var)

                for value in np.unique(illustrative_var):

                    selected = np.where(illustrative_var == value)

                    plt.scatter(X_projected[selected, d1], X_projected[selected, d2], alpha=alpha, label=value)

                plt.legend()



            # Display the labels on the points

            if labels is not None:

                for i,(x,y) in enumerate(X_projected[:,[d1,d2]]):

                    plt.text(x, y, labels[i],

                              fontsize='14', ha='center',va='center') 

                

            # Define the limits of the chart

            boundary = np.max(np.abs(X_projected[:, [d1,d2]])) * 1.1

            plt.xlim([-boundary,boundary])

            plt.ylim([-boundary,boundary])

        

            # Display grid lines

            plt.plot([-100, 100], [0, 0], color='grey', ls='--')

            plt.plot([0, 0], [-100, 100], color='grey', ls='--')



            # Label the axes, with the percentage of variance explained

            plt.xlabel('PC{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))

            plt.ylabel('PC{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))



            plt.title("Projection of points (on PC{} and PC{})".format(d1+1, d2+1))

            #plt.show(block=False)

   

def display_scree_plot(pca):

    '''Display a scree plot for the pca'''



    scree = pca.explained_variance_ratio_*100

    plt.bar(np.arange(len(scree))+1, scree)

    plt.plot(np.arange(len(scree))+1, scree.cumsum(),c="red",marker='o')

    plt.xlabel("Number of principal components")

    plt.ylabel("Percentage explained variance")

    plt.title("Scree plot")

    plt.show(block=False)



def append_class(df, class_name, feature, thresholds, names):

    '''Append a new class feature named 'class_name' based on a threshold split of 'feature'.  Threshold values are in 'thresholds' and class names are in 'names'.'''

    

    n = pd.cut(df[feature], bins = thresholds, labels=names)

    df[class_name] = n



def plot_dendrogram(Z, names, figsize=(10,25)):

    '''Plot a dendrogram to illustrate hierarchical clustering'''



    plt.figure(figsize=figsize)

    plt.title('Hierarchical Clustering Dendrogram')

    plt.xlabel('distance')

    dendrogram(

        Z,

        labels = names,

        orientation = "left",

    )

    #plt.show()



def addAlpha(colour, alpha):

    '''Add an alpha to the RGB colour'''

    

    return (colour[0],colour[1],colour[2],alpha)



def display_parallel_coordinates(df, num_clusters):

    '''Display a parallel coordinates plot for the clusters in df'''



    # Select data points for individual clusters

    cluster_points = []

    for i in range(num_clusters):

        cluster_points.append(df[df.cluster==i])

    

    # Create the plot

    fig = plt.figure(figsize=(12, 15))

    title = fig.suptitle("Parallel Coordinates Plot for the Clusters", fontsize=18)

    fig.subplots_adjust(top=0.95, wspace=0)



    # Display one plot for each cluster, with the lines for the main cluster appearing over the lines for the other clusters

    for i in range(num_clusters):    

        plt.subplot(num_clusters, 1, i+1)

        for j,c in enumerate(cluster_points): 

            if i!= j:

                pc = parallel_coordinates(c, 'cluster', color=[addAlpha(palette[j],0.2)])

        pc = parallel_coordinates(cluster_points[i], 'cluster', color=[addAlpha(palette[i],0.5)])



        # Stagger the axes

        ax=plt.gca()

        for tick in ax.xaxis.get_major_ticks()[1::2]:

            tick.set_pad(20)        





def display_parallel_coordinates_centroids(df, num_clusters):

    '''Display a parallel coordinates plot for the centroids in df'''



    # Create the plot

    fig = plt.figure(figsize=(12, 5))

    title = fig.suptitle("Parallel Coordinates plot for the Centroids", fontsize=18)

    fig.subplots_adjust(top=0.9, wspace=0)



    # Draw the chart

    parallel_coordinates(df, 'cluster', color=palette)



    # Stagger the axes

    ax=plt.gca()

    for tick in ax.xaxis.get_major_ticks()[1::2]:

        tick.set_pad(5)    
%config IPCompleter.greedy=True

import pandas as pd

import numpy as nm

import warnings



from bokeh.io import output_file, output_notebook

from bokeh.plotting import figure, show

from bokeh.models import ColumnDataSource

from bokeh.layouts import row, column, gridplot

from bokeh.models.widgets import Tabs, Panel

import scipy.stats as sst



warnings.filterwarnings("ignore")

data = pd.read_csv("../input/world-food-facts/en.openfoodfacts.org.products.tsv", delimiter='\t',dtype='unicode')

data.describe()
def filterSet(data,term):

    ##selecting right columns

    data1= data.filter(regex=term)

    ##removing all rows without values

    data1.dropna(axis = 0, how = 'all', inplace = True)

    ##if column contains less than 1000 observations remove it

    dropColumns(data1,1000)

    return data1
def dropColumns(data,count):

    for column in data:

        if data[column].count()<=1000:

            del data[column]

    return data
def testDropColumn(data,count):

    print("Dataset has "+str(len(data.columns))+" columns before removal")

    data1=dropColumns(data,count)

    data2=data.dropna(thresh=count,how='all',axis=1)

    if(len(data1.columns)==len(data2.columns)):

        print( "Test is ok")

    else:

        print("Test did not succeed length by dropna= "+str(len(data2.columns))+" length by function "+ str(len(data2.columns)))

    
def removeDuplicates(data, term):

    data1=data

    ##remove duplicates rows depending on selected column. I use it to delete duplicates for product name

    data1['add']= term

    data1=data1.drop_duplicates(subset='add', keep="first")

    del data1['add']

    return data1
def removeDuplicatesTest(data,term,result):

    res=removeDuplicates(data,term)

    if res.shape[0]==result.shape[0]:

        return "test is OK"

    else: 

        return res.shape[0]
duplicated=data.drop_duplicates( subset='product_name',keep="first")

prepared=data.drop('product_name', axis=1)

removeDuplicatesTest(prepared,data['product_name'],duplicated)
def showData(column,data2):

    import scipy.stats as sst

    ##preprocessing dataset

    data2=filterSet(data,column)

    data2=removeDuplicates(data2,data['product_name'])

    

    ##convert object to float and keep positive values

    data2[data2.columns]=data2[data2.columns].apply(pd.to_numeric, errors='coerce')

    data2=data2.abs()

    

    ##add column with qualitative data

    data2['main_category']=data['main_category']

    data2 = data2[pd.notnull(data2['main_category'])]

    

    ##fill NAN with 0

    data2.fillna(0, inplace=True)

    ##format qualitative column

    data3=data2[data2["main_category"].str.contains('en')]

    data3["main_category"]=data3["main_category"].str[3:]

    

    ##group by product category having count >1600, we want to have groups with 1600 observations

    data4=data3.groupby("main_category").filter(lambda x: len(x) > 1600)

    categories = data4['main_category'].unique()

    groups = []

    ##create list that contains lists observations for each category

    for m in categories:

        groups.append(data4[data4['main_category']==m][column])

        

    ##drawing boxplots

    plt.boxplot(groups,labels=categories,vert=False,showfliers=False )

    plt.title(column)

    plt.xlabel('Measure units')

    plt.ylabel('Food categories')

    plt.show()

    

    

    ##F statistic calculation

    count=0

    sum=0

    ##list with mean for each column

    means=[]

    for i in groups:

        mn=i.mean()

        means.append(mn)

        count+=len(i)

        for u in i:

            sum+=u

    ##Grand mean

    MMean=(sum/count)

    

    ##Estimated sum of squares

    ESS=0

    ##Residual sum of squares

    RSS=0

    ##Degree of freedom TSS

    DfT=count-1

    ##Degree of freedom ESS

    DfE=len(groups)-1

    ##Degree of freedom RSS

    DfR=DfT-DfE

    ccount=0

    for y in means:

        ESS+=(y-MMean)**2*len(groups[ccount])

        for ii in groups[ccount]:

            RSS+=(ii-y)**2

        ccount+=1

    ##Total sum of squares

    TSS=0

    for h in groups:

        for j in h:

            TSS+=(j-MMean)**2

    print("TTS: "+str(TSS)+ " ESS: "+ str(ESS)+ " Cooficient of Determination " +str(ESS/TSS)+ ' RSS '+ str(RSS))

    print("F value : "+str((ESS/DfE)/(RSS/DfR)))

    import scipy.stats as stats

    print("Critical Value: "+str(stats.f.ppf(q=0.99, dfn=ESS/DfE, dfd=RSS/DfR)))

    print("-----------------")

    ##direct calculation using scipy library

    if len(groups)==5:

        print(stats.f_oneway(groups[0],groups[1],groups[2] ,groups[3],groups[4]  ))

        print("----------Testing Assumptions----------------")

        ##print(str(sst.shapiro(groups[0])[1])+" "+str(sst.shapiro(groups[1])[1])+" "+str(sst.shapiro(groups[2])[1])+" "+str(sst.shapiro(groups[3])[1])+" "+str(sst.shapiro(groups[4])[1]))

        ##testing fro normality

        checko=True

        for u in groups:

            if sst.shapiro(u)[1]<0.05:

                checko=False

        if checko ==True:

            print("Normality check with Shapiro test is satisfied")

        else:

            print("Normality check with Shapiro test is not satisfied")

            ##testing for homoskedasticity

        

    print("Homoskedasticity check with Bartlett test: p="+ str(sst.bartlett(groups[0],groups[1],groups[2] ,groups[3],groups[4] )[1]))

    
def show2Variab(data):

    ##preproces data

    data2=data

    data2.dropna(axis = 0, how = 'all', inplace = True)

    data2[(data2.T != 0).any()]

    data2[data2.columns]=data2[data2.columns].apply(pd.to_numeric, errors='coerce')

    data2=data2.abs()

    data2.fillna(0, inplace=True)

    ##remove outliers, we remove 1 percent of largest observations 

    return removeOutliers(data2)
def removeOutliers(data2):

    for column in data2:

        q = data2[column].quantile(0.99)

        data2=data2[data2[column] < q]

    return data2
dataa=pd.DataFrame(data, columns = ['energy_100g','sugars_100g'])

dataa=show2Variab(dataa)

dataa.describe()
import plotly.express as px

fig = px.scatter(dataa, x="energy_100g", y="sugars_100g", title="Energy/Sugars")

fig.show()

dataa=pd.DataFrame(data, columns = ['proteins_100g','sugars_100g'])

dataa=show2Variab(dataa)

dataa.describe()
fig = px.scatter(dataa, x="proteins_100g", y="sugars_100g",title='Proteins/Sugars')

fig.show()



dataa=pd.DataFrame(data, columns = ['proteins_100g','energy_100g'])

dataa=show2Variab(dataa)

dataa.describe()
fig = px.scatter(dataa, x="proteins_100g", y="energy_100g",title='Proteins/Energy')

fig.show()

import matplotlib.pyplot as plt

showData('proteins_100g',data)

showData('fat_100g',data)

showData('energy_100g',data)

showData('sugars_100g',data)
data1= filterSet(data,'100g')

data1=data1.drop(["nutrition-score-uk_100g","nutrition-score-fr_100g"], axis=1)

print(data1.columns)
data1=removeDuplicates(data1,data['product_name'])
data1[data1.columns]=data1[data1.columns].apply(pd.to_numeric, errors='coerce')

data1.dropna(axis=0, thresh=19, inplace=True)
data1= data1.abs()
data1.fillna(0, inplace=True)

from sklearn.preprocessing import StandardScaler

# Standardize the data

scaler = StandardScaler()

X_scaled = scaler.fit_transform(data1)
from sklearn.decomposition import PCA



num_components = 10

# Create the PCA model

pca = PCA(n_components=num_components)



# Fit the model with the standardised data

pca.fit(X_scaled)

nwD=pca.transform(X_scaled)

pcs = pca.components_ 

pca.explained_variance_ratio_.cumsum()


display_scree_plot(pca) 
%run functions

# Generate a correlation circle

pca.components_=np.around(pca.components_, decimals=4)

pcs=pca.components_

display_circles(pcs, num_components, pca, [(0,1)], labels = np.array(data1.columns),)
display_circles(pcs, num_components, pca, [(2,3)], labels = np.array(data1.columns),)
num_components=4

pca = PCA(n_components=num_components)

# Fit the model with the standardised data

pca.fit(X_scaled)

nwD=pca.transform(X_scaled)

pcs = pca.components_ 
principalDf = pd.DataFrame(nwD, columns = range(1,num_components+1))

from sklearn.cluster import KMeans

# Run a number of tests, for 1, 2, ... num_clusters

num_clusters = 9

kmeans_tests = [KMeans(n_clusters=i, init='random', n_init=10) for i in range(1, num_clusters)]

score = [kmeans_tests[i].fit(principalDf).score(principalDf) for i in range(len(kmeans_tests))]
# Plot the curve

plt.plot(range(1, num_clusters),score)

plt.xlabel('Number of Clusters')

plt.ylabel('Score')

plt.title('Elbow Curve')

plt.show()
# Create a k-means clustering model

kmeans = KMeans(init='random', n_clusters=4, n_init=10)



# Fit the data to the model

kmeans.fit(principalDf)



# Determine which clusters each data point belongs to:

clusters =  kmeans.predict(principalDf)
# Add cluster number to the original data

X_clustered = pd.DataFrame(principalDf, columns=principalDf.columns, index=principalDf.index)

X_clustered['cluster'] = clusters
%run functions

# Display parallel coordinates plots, one for each cluster

display_parallel_coordinates(X_clustered, 4)