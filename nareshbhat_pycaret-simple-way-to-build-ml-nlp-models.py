%%capture
!pip install pycaret
from pycaret.regression import *
import pandas as pd
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
test.head()
reg = setup(data = train,target = 'SalePrice', numeric_imputation = 'mean',normalize = True,
             ignore_features = ['Alley','PoolQC','MiscFeature','Fence','FireplaceQu','Utilities'],pca=True,
    pca_method='linear',pca_components=30,silent = True,session_id = 3650)
compare_models(exclude = ['tr'] , turbo = True) 
cb = create_model('catboost')
interpret_model(cb)
prediction = predict_model(cb, data = test)
output_reg = pd.DataFrame({'Id': test.Id, 'SalePrice': prediction.Label})
output_reg.to_csv('submission.csv', index=False)
output_reg.head()
import pandas as pd 
from pycaret.classification import *
train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')
test.head()
classification_setup = setup(data= train, target='Survived',remove_outliers=True,normalize=True,normalize_method='robust',
                            ignore_features= ['Name'], silent = True,session_id = 6563)
compare_models(exclude = ['lda'])
dt = create_model('dt')
plot_model(estimator = dt, plot = 'auc')
plot_model(estimator = dt, plot = 'feature')
rd = create_model('ridge');      
lgm  = create_model('lightgbm');            

#blending 3 models
blend = blend_models(estimator_list=[lgm,dt,rd])
optimize_threshold(lgm, true_negative = 500, false_negative = -2000)
plot_model(estimator = blend, plot = 'confusion_matrix')
pred = predict_model(blend, data = test)
output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': pred.Label})
output.to_csv('submission.csv', index=False)
output.head()
import pandas as pd
from pycaret.clustering import *
data = pd.read_csv('../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')
data.head()
exp_clu = setup(data)
kmeans = create_model('kmeans')
print(kmeans)

plot_model(kmeans,plot = 'elbow')
plot_model(kmeans,plot = 'distance')
plot_model(kmeans,plot = 'tsne')
plot_model(kmeans,plot = 'silhouette')
plot_model(kmeans,plot = 'distribution')
kmeans_df = assign_model(kmeans)
kmeans_df.head()
hierarchical = create_model('hclust')
plot_model(hierarchical,plot='cluster',label = True )
hierarchical_df = assign_model(hierarchical)
hierarchical_df.head()
from pycaret.nlp import *
import pandas as pd
data = pd.read_csv('../input/spam-text-message-classification/SPAM text message 20170820 - Data.csv')
data.head()
nlp = setup(data = data, target = 'Message', session_id = 1)
lda = create_model('lda', multi_core = True)
lda_data = assign_model(lda)
lda_data.head()
evaluate_model(lda)
from pycaret.classification import *
model = setup(data = lda_data, target = 'Category',ignore_features=['Message','Dominant_Topic','Perc_Dominant_Topic'])
compare_models()
