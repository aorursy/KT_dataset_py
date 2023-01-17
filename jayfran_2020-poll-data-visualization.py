# Read the csv file

import pandas as pd 

file = '../input/2020_data_pct.csv'

df_2020 = pd.read_csv(file,encoding='latin-1',index_col=0)



# Drop the '%' signs

for col in list(df_2020.columns):

    df_2020[col] = [x.strip('%') for x in df_2020[col]]



# Swap index and columns

df_2020 = df_2020.transpose().astype(int)

df_2020.info() 
# Peek at dataframe containing voting percentages for three

# hypothetical scenarios for 2020 general election

df_2020
# Compute the spreads of democratic candidates over Trump

# spread = (% Dem - % Trump)

# Positive values indi

biden_trump_sprd = df_2020['1. Joe Biden'] - df_2020['1. Donald Trump'] 

sanders_trump_sprd = df_2020['2. Bernie Sanders'] - df_2020['2. Donald Trump'] 

beto_trump_sprd = df_2020['3. Beto OÕRourke'] - df_2020['3. Donald Trump'] 

# combine these into an new df with spreads instead

df_spread = pd.DataFrame(data=[biden_trump_sprd,sanders_trump_sprd,beto_trump_sprd],index=['biden-trump','sanders-trump','beto-trump']).transpose()

df_spread
# Create alternate dataset to incorporate the uncertainty of voters

# confidence score = (% Dem - % Trump) / (% "Too early to say")

# There are some instances where the scale of this score is very wide (>700),

#    in which case I may present it as a log scale.

df_conf = pd.DataFrame()

df_conf['biden-trump'] = 100*df_spread['biden-trump']/df_2020['1. Too early for me to say']

df_conf['sanders-trump'] = 100*df_spread['sanders-trump']/df_2020['2. Too early for me to say']

df_conf['beto-trump'] = 100*df_spread['beto-trump']/df_2020['3. Too early for me to say']

df_conf
# quick visual of the three matchups across 38 demographic groups

import matplotlib

import matplotlib.pyplot as plt 

%matplotlib inline

matplotlib.rcParams.update({'font.size': 14,'figure.figsize':(14,5),'axes.grid':'true',

                            'grid.linestyle':':','grid.linewidth':1.5,'grid.alpha':.5})

df_spread.plot()

plt.title('Spread of democratic candidate over Trump across 38 demographic groups')

plt.xlabel('Demographic Groups')

plt.ylabel('Spread of Dem. Over Trump')

plt.show()



df_conf.plot()

plt.title('Confidence Score of democratic candidate over Trump across 38 demographic groups')

plt.xlabel('Demographic Groups')

plt.ylabel('Conf. Score of Dem. Over Trump')

plt.show()



df_conf.plot()

plt.title('log(Confidence Score) of democratic candidate over Trump across 38 demographic groups')

plt.xlabel('Demographic Groups')

plt.ylabel('log(Conf. Score of Dem. Over Trump')

plt.yscale('symlog')

plt.show()
# A function to plot each category using either: spreads, conf. score, or log(conf. score)

# Takes parameters: the dataframe to use, a string = 'spread','conf','log'

def plot_categories(df,plot_type='spread'): 



    # import and set parameters

    import numpy as np

    import matplotlib

    import matplotlib.pyplot as plt 

    from IPython.display import display, HTML

    %matplotlib inline

    matplotlib.rcParams.update({'font.size': 14,'figure.figsize':(6,4),'axes.grid':'true',

                                'grid.linestyle':':','grid.linewidth':1.5,'grid.alpha':1})

    

    # a little set up to help loop through plotting each category

    demo_cats = ['Total','Gender','Age','Generation','Race/Ethnicity','Region',

                   'Locale','Education','Income','Party ID','Ideology']

    demo_start_ind = [0,1,3,7,12,17,21,24,27,29,33]    # where each category begins

    

    fig = plt.figure()

    fig.subplots_adjust(hspace=0.4, wspace=0.4)



    # declare the title and labels based on the plot_type parameter

    if plot_type=='spread':

        plot_title = 'Spread: '

        plot_ylabel = 'Spread of Dem. Over Trump'

    elif plot_type=='conf':

        plot_title = 'Confidence Score: '

        plot_ylabel = 'Conf. Score of Dem. Over Trump'

    elif plot_type=='log':

        plot_title = 'log(Conf Score): '

        plot_ylabel = 'log(Conf. Score) of Dem. Over Trump'

    else: print('plot_type:'+plot_type+' not recognized.')

        

    

    # loop through categories

    for i in range(len(demo_start_ind)):

        # all but the last plot

        if i<len(demo_start_ind)-1:

            # take a slice of the dataset for a given category

            start = demo_start_ind[i]

            stop = demo_start_ind[i+1]

            df.iloc[start:stop,:].plot(title=plot_title+demo_cats[i],kind='bar')

            plt.ylabel(plot_ylabel)  

            plt.xticks(rotation=60)

            #plt.ylim(-50,50)

            if plot_type == 'log':

                plt.yscale('symlog')

                plt.ylim(-1000,1000)

        

        # the last plot

        else:

            df_spread.iloc[demo_start_ind[i]:,:].plot(title=plot_title+demo_cats[i],kind='bar')

            plt.ylabel(plot_ylabel)    

            #df_conf.iloc[demo_start_ind[i]:,:].plot(title=demo_cats[i],kind='bar')

            #plt.ylabel('Conf. Score of Dem. Over Trump')    

    plt.show()
# Spread version      *** Click the sidebar below 'In' to expand figure window

plot_categories(df_spread,'spread')
# Confidence Score version     *** Click the sidebar below 'In' to expand figure window

plot_categories(df_conf,'conf')
# Confidence Score version            *** Click the sidebar below 'In' to expand figure window

plot_categories(df_conf,'log')
df_spread.loc[['Male','Baby Boomers','Asian'],:]
# Function that runs KMeans algorithm, returns a modified dataframe with assignments,

# optionally prints summary of clusters, and visualizes result as 2D or 3D scatter plot



def run_kmeans(df, k=5, print_clusters=True, plot='2D',return_df=False):

    # import pandas, KMeans

    import pandas as pd

    from sklearn.cluster import KMeans

    

    # create kmeans object and fit to data

    kmeans = KMeans(n_clusters=k)

    kmeans.fit(df)

    # save location of cluster centroids learned by kmeans object

    centroids = kmeans.cluster_centers_

    

    # save the cluster assignments as a series

    s = pd.Series(kmeans.labels_)

    # add the values of this series to the df

    df = df.assign(cluster=s.values)

    # create groupby object to view by cluster

    grouped = df.groupby('cluster').mean()

    

    # Optional feature to print cluster detail

    if print_clusters:

        

        # print cluster summary

        print('-----------------------------------------------------')

        print('KMEANS SUMMARY - MEAN SPREADS')

        print('-----------------------------------------------------')

        print(grouped)

        print()

        # print cluster summary

        print('CLUSTER MEMBER DETAILS')

        print('--------------------------------------------------------------------------')

        for i in range(len(centroids)):

            print('Cluster: '+str(i))

            print('--------------------------------------------------------------------------')

            print(df[df['cluster']==i])

            print()

            print('--------------------------------------------------------------------------')

    

    # import: mcols for custom colormap, matplotlib

    import matplotlib.colors as mcols

    # define a colormap for plots using a list of custom colors

    cmap = mcols.ListedColormap(['g','cyan','r','orange','purple'])

    

    # 2-D plots

    if plot=='2D':

        def plot_scatter(x_index,y_index):

            import matplotlib.pyplot as plt

            plt.figure(figsize=(6,6))

            # first plot the spreads as a scatter plot

            plt.scatter(df.iloc[:,x_index], df.iloc[:,y_index], 

                        cmap=cmap, c=kmeans.labels_,alpha=.3)

            # add the centroid locations for each cluster

            plt.scatter(centroids[:,x_index], centroids[:,y_index],marker='x', 

                        s=200, linewidths=10,cmap=cmap,c=range(k), zorder=10)

            plt.xlim((-80,80))

            plt.ylim((-80,80))

            plt.title("K-means Clustering on Demographic Groups")

            plt.xlabel(df.columns[x_index])

            plt.ylabel(df.columns[y_index])

            

            # draw diagonal line from (70, 90) to (90, 200)

            plt.annotate("",

              xy=(-100, -100), xycoords='data',

              xytext=(100, 100), textcoords='data',

              arrowprops=dict(arrowstyle="-",

                              connectionstyle="arc3,rad=0."),)

                        

            # Add diagonal line for reference

            plt.plot([-100, 100], [-100, 100], color='black', linestyle='--', linewidth=1)

            plt.show()

        

        # Repeat for different candidate comparisons

        plot_scatter(0,1)

        plot_scatter(0,2)

        plot_scatter(1,2)



    # 3-D plots

    if plot=='3D':

        import matplotlib.pyplot as plt

        import matplotlib

        from mpl_toolkits.mplot3d import Axes3D

        %matplotlib inline

        matplotlib.rcParams.update({'font.size': 13})

        

        threedee = plt.figure(figsize=(15,15)).gca(projection='3d')

        threedee.scatter(df.iloc[:,0],df.iloc[:,1],df.iloc[:,2],c=kmeans.labels_,cmap=cmap,alpha=.3)

        threedee.scatter(centroids[:,0],centroids[:,1], centroids[:,2],

                marker='o', s=169, linewidths=15,cmap=cmap,c=range(k))

        threedee.set_xlabel(df.columns[0])

        threedee.set_ylabel(df.columns[1])

        threedee.set_zlabel(df.columns[2])

        threedee.set_title("K-means Clustering on Demographic Groups")

        plt.show()

        

    if return_df:

        return df

    else: return
# Function that runs hierarchical clustering algorithm, returns a modified dataframe with assignments,

# optionally prints summary of clusters, and visualizes result as 2D or 3D scatter plot



def run_heirarchy(df, n=5, print_clusters=True, plot='2D',plot_dendrogram=False,return_df=False):

    # import pandas

    import pandas as pd

    

    # import hierarchical clustering libraries

    import scipy.cluster.hierarchy as sch

    from sklearn.cluster import AgglomerativeClustering

    

    # plot dendrogram

    if plot_dendrogram:

        import matplotlib.pyplot as plt

        plt.figure(1, figsize=(10, 10))

        plt.title('Hierarchical Clustering of Demographic Groups - Dendogram')

        dendrogram = sch.dendrogram(sch.linkage(df, method='ward'),labels=df.index,color_threshold=55,

                                    leaf_rotation=0,leaf_font_size=10,orientation='left')

    

    # create clusters

    hc = AgglomerativeClustering(n_clusters=n, affinity = 'euclidean', linkage = 'ward')

    # save clusters for chart

    clusters = hc.fit_predict(df)



    # save the cluster assignments as a series

    s = pd.Series(clusters)

    # add the values of this series to the df

    df = df.assign(cluster=s.values)

    # create groupby object to view by cluster

    grouped = df.groupby('cluster').mean()

    

    # Optional feature to print cluster detail

    if print_clusters:  

        # print cluster summary

        print('-----------------------------------------------------')

        print('HIERARCHICAL CLUSTERING SUMMARY - MEAN SPREADS')

        print('-----------------------------------------------------')

        print(grouped)

        print()

        # print cluster summary

        print('CLUSTER MEMBER DETAILS')

        print('--------------------------------------------------------------------------')

        for i in range(n):

            print('Cluster: '+str(i))

            print('--------------------------------------------------------------------------')

            print(df[df['cluster']==i])

            print()

            print('--------------------------------------------------------------------------')

    

    # import: mcols for custom colormap, matplotlib

    import matplotlib.colors as mcols

    # define a colormap for plots using a list of custom colors

    cmap = mcols.ListedColormap(['g','cyan','r','orange','purple'])

    

    # 2-D plots

    if plot=='2D':

        def plot_scatter(x_index,y_index):

            import matplotlib.pyplot as plt

            plt.figure(figsize=(6,6))

            # plot the spreads as a scatter plot

            plt.scatter(df.iloc[:,x_index], df.iloc[:,y_index], 

                        cmap=cmap, c=df['cluster'],alpha=.5)

            plt.xlim((-80,80))

            plt.ylim((-80,80))

            plt.title("Hierarchical Clustering on Demographic Groups")

            plt.xlabel(df.columns[x_index])

            plt.ylabel(df.columns[y_index])

            

            # Add diagonal line for reference

            plt.plot([-100, 100], [-100, 100], color='black', linestyle='--', linewidth=1)

            plt.show()

        

        # Repeat for different candidate comparisons

        plot_scatter(0,1)

        plot_scatter(0,2)

        plot_scatter(1,2)



    # 3-D plots

    if plot=='3D':

        import matplotlib.pyplot as plt

        import matplotlib

        from mpl_toolkits.mplot3d import Axes3D

        %matplotlib inline

        matplotlib.rcParams.update({'font.size': 13})

        # 

        threedee = plt.figure(figsize=(15,15)).gca(projection='3d')

        threedee.scatter(df.iloc[:,0],df.iloc[:,1],df.iloc[:,2],c=df['cluster'],cmap=cmap,alpha=.4)

        #threedee.scatter(centroids[:,0],centroids[:,1], centroids[:,2],

        #        marker='o', s=169, linewidths=15,cmap=cmap,c=range(k))

        threedee.set_xlabel(df.columns[0])

        threedee.set_ylabel(df.columns[1])

        threedee.set_zlabel(df.columns[2])

        threedee.set_title("Hierarchical Clustering on Demographic Groups")

        plt.show()

    

    if return_df:

        return df

    else: return
# Run the kmeans/heirarchical clustering algorithms on a spread basis

# Assignments are nearly (if not perfectly) identical between methods



run_heirarchy(df_spread,plot='none')
# View dendrogram from heirarchical clustering

run_heirarchy(df_spread,print_clusters=False,plot='none',plot_dendrogram=True)
# 3D Visualizarion of the Clusters 

df_km = run_heirarchy(df_spread,print_clusters=False,plot='3D')
# 2D Visualization of the Clusters 

df_hc = run_heirarchy(df_spread,print_clusters=False,plot='2D',return_df=True)
# A closer look at the groups with < 20 pt spread (Biden vs Sanders)



# import: mcols for custom colormap, matplotlib

import matplotlib.pyplot as plt

import matplotlib.colors as mcols

%matplotlib inline

matplotlib.rcParams.update({'font.size': 14,'figure.figsize':(1,1),'axes.grid':'true',

                                'grid.linestyle':':','grid.linewidth':1.5,'grid.alpha':1})





# define a colormap for plots using a list of custom colors

cmap = mcols.ListedColormap(['g','cyan','r','orange','purple'])

        

plt.figure(figsize=(6,6))

# plot the spreads as a scatter plot

plt.scatter(df_hc.iloc[:,0], df_hc.iloc[:,1], 

            cmap=cmap, c=df_hc['cluster'],alpha=.5)

plt.xlim((-10,10))

plt.ylim((-10,10))

plt.title("Hierarchical Clustering on Demographic Groups")

plt.xlabel(df_hc.columns[0])

plt.ylabel(df_hc.columns[1])



labels = df_hc.index

for label, x, y in zip(labels, df_hc.iloc[:, 0], df_hc.iloc[:, 1]):

    plt.annotate(

        label,

        xy=(x, y), 

        xytext=(-20, 20),

        textcoords='offset points', ha='right', va='bottom',

        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),

        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))



# Add diagonal line for reference

plt.plot([-100, 100], [-100, 100], color='black', linestyle='--', linewidth=1)

plt.show()





# Not sure why this is coming out so tiny...