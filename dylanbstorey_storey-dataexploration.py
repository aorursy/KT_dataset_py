import pandas as pd

import numpy as np

import seaborn as sns







%matplotlib inline

#sns.set_context('poster')
data = pd.read_csv('../input/train.csv')

features = list(data.columns)

features.remove('id')

features.remove('loss')

cat_features = [x for x in features if x.find('cat') != -1]

cont_features = [x for x in features if x.find('cont') != -1]
# No Missing Data ! A MIRACLE

data.isnull().any().any()
correlationMatrix = data.copy()

correlationMatrix.drop(cat_features+['id','loss'],inplace=True,axis=1)

correlationMatrix['logLoss'] = np.log(data.loss)



correlationMatrix = correlationMatrix.corr().abs()

map = sns.clustermap(correlationMatrix,annot=True,annot_kws={"size": 10})

sns.plt.setp(map.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)

sns.plt.show()
# I think we all know what these look like - I'm not uploading them.

# cat_dat = data.copy()

# cat_dat.drop(cont_features,inplace=True,axis = 1)

# for i in cat_features:

#     try:

#         sns.boxplot(data=cat_dat,x=i,y=np.log(data.loss))

#         sns.plt.show()

#     except:

#         print('{} failed for some reason'.format())
# Same thing here, if you want to draw them feel free to fork and run

# for i in cont_features:

#     try:

#         sns.distplot(data[i])

#         sns.plt.show()

#     except:

#         print("{} failed for some reason".format(i))