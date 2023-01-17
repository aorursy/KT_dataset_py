import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns



from sklearn.decomposition import PCA

from sklearn.manifold import TSNE
df = pd.read_csv('../input/covtype.csv', nrows=7500)

df.head()
df.info()    # No missing values and all variables are of type int64
df['Cover_Type'].value_counts()    # Cover types encoded by integers
class_dict = {1:'Spruce/Fir',

              2:'Lodgepole Pine',

              3:'Ponderosa Pine',

              4:'Cottonwood/Willow',

              5:'Aspen',

              6:'Douglas-fir',

              7:'Krummholz'

             }



# Map the integers to their names



df['Cover_Type_Name'] = df['Cover_Type'].map(class_dict)



# One-hot encoding

df_updated = pd.concat([df, pd.get_dummies(df['Cover_Type_Name'], prefix='Type')],

                   axis=1)
# Plot all features



all_corr = df_updated.corr()

all_corr.dropna(axis=[0,1], how='all', inplace=True)



plt.rcParams['figure.figsize'] = [16,12]

mask = np.zeros_like(all_corr,dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



sns.heatmap(all_corr,cmap = sns.diverging_palette(220, 10, as_cmap=True),center=0,mask=mask)
# Plot the top correlated features (abs to convert -ve correlations accordingly) by Cover Type



plt.rcParams['figure.figsize'] = [16,2]



for cov in class_dict.values():

    

    cov_str = 'Type_' + cov

    type_corr = all_corr[[cov_str]].apply(abs).sort_values(by=cov_str, ascending=False)

    

    sns.heatmap(type_corr.iloc[1:11,:].transpose(), 

                annot=True, 

                cmap = sns.diverging_palette(220, 10, as_cmap=True),

                center=0,

                linewidths=0.5)

    plt.yticks([])

    plt.title('Top Correlated Features with {}'.format(cov_str)),

    plt.show()

    

#all_corr[['Type_Aspen']].apply(abs).sort_values(by='Type_Aspen', ascending=False).transpose()
def boxplot_sorted(df, by, column, rot=0):

    

    # use dict comprehension to create new dataframe from the iterable groupby object

    # each group name becomes a column in the new dataframe

    

    df2 = pd.DataFrame({col:vals[column] for col, vals in df.groupby(by)})

    

    # find and sort the median values in this new dataframe

    

    meds = df2.median().sort_values()

    

    # use the columns in the dataframe, ordered sorted by median value

    # return axes so changes can be made outside the function

    return df2[meds.index].boxplot(rot=rot, return_type="axes")
plt.rcParams['figure.figsize'] = [16,8]



location_vars = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',

                 'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',

                 'Horizontal_Distance_To_Fire_Points'

                ]



for var in location_vars:

    axes = boxplot_sorted(df_updated, by="Cover_Type_Name", column=var)

    axes.set_title("{} by Cover_Type".format(var))

    plt.show()
plt.rcParams['figure.figsize'] = [12,24]



for ind, col in enumerate(location_vars):

    plt.subplot(len(location_vars), 2, 2*ind + 1)

    df_updated[col].plot.kde()

    plt.xlim(left=df_updated[col].min())

    plt.title(col)

    

    plt.subplot(len(location_vars), 2, 2*ind + 2)

    df_updated[col].hist()

    plt.title(col)

    

plt.tight_layout()

plt.show()
# Utility function to visualize the outputs of PCA and t-SNE



def fashion_scatter(x, labels):

    # choose a color palette with seaborn.

    num_classes = len(np.unique(labels.values))

    palette = np.array(sns.color_palette("hls", num_classes))



    # create a scatter plot.

    f = plt.figure(figsize=(8, 8))

    ax = plt.subplot(aspect='equal')

    

    for i in range(num_classes):

        index = labels.loc[labels == i + 1].index

        sc = ax.scatter(x[index,0], x[index,1], lw=0, s=30, c=palette[i],label=class_dict[i+1])

    plt.xlim(-25, 25)

    plt.ylim(-25, 25)

    ax.axis('tight')

    ax.legend()



    return f, ax, sc
# PCA with 2 principal components



pca = PCA(n_components=2)



pca_result = pca.fit_transform(df_updated.loc[:,location_vars])



print("Total explained variance ratio (based on 2 components) = {:.2f}".format(pca.explained_variance_ratio_.sum()))



pca_df = pd.DataFrame(pca_result,columns=['PCA1','PCA2'])



pca_df = pd.concat([pca_df,df_updated['Cover_Type']],axis=1)



fashion_scatter(pca_df.loc[:,['PCA1','PCA2']].values, pca_df['Cover_Type'])
tsne = TSNE(random_state=234).fit_transform(df_updated.loc[:,location_vars])



fashion_scatter(tsne, pca_df['Cover_Type'])