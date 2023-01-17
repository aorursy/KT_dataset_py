#Nomes dos integrantes
#Fernanda Piva - 20.83990-0
#Rodrigo Franciozi - 20.83984-7
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from plotly.offline import init_notebook_mode, iplot, plot
import plotly as py
init_notebook_mode(connected=True)
import plotly.graph_objs as go

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Reading the sellers dataset
df_sellers = pd.read_csv("../input/brazilian-ecommerce/olist_sellers_dataset.csv")

# Reading the costumers dataset
df_costumers = pd.read_csv("../input/brazilian-ecommerce/olist_customers_dataset.csv")

# Reading the orders datasets
df_orders = pd.read_csv("../input/brazilian-ecommerce/olist_orders_dataset.csv")
df_order_items = pd.read_csv("../input/brazilian-ecommerce/olist_order_items_dataset.csv")
df_order_payments = pd.read_csv("../input/brazilian-ecommerce/olist_order_payments_dataset.csv")
df_order_reviews = pd.read_csv("../input/brazilian-ecommerce/olist_order_reviews_dataset.csv")

# Reading the products dataset
df_products = pd.read_csv("../input/brazilian-ecommerce/olist_products_dataset.csv")

# Reading the localization dataset
df_localizations = pd.read_csv("../input/brazilian-ecommerce/olist_geolocation_dataset.csv")

# Reading the category name translator
df_translator = pd.read_csv("../input/brazilian-ecommerce/product_category_name_translation.csv")
df_products
# Reading the leads dataset 
df_leads = pd.read_csv('../input/marketing-funnel-olist/olist_marketing_qualified_leads_dataset.csv')
df_leads.head(10)
# Reading the leads dataset 
df_closed_deals = pd.read_csv('../input/marketing-funnel-olist/olist_closed_deals_dataset.csv')
df_closed_deals.head(10)
df_text = df_order_reviews[["review_score", "review_comment_title"]].dropna()
df_text 
from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer()

X_counts = count_vect.fit_transform(df_text["review_comment_title"].to_list())
X_counts.shape
print(X_counts)
from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer()
X_tfidf = tfidf_transformer.fit_transform(X_counts)
X_tfidf.shape
print(X_tfidf)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_tfidf, df_text['review_score'], test_size = 0.15, random_state = 42)
print('X Train shape:' , X_train.shape)
print('Y Train shape:', y_train.shape)
print('X test shape: ', X_test.shape)
print('y test shape:', y_test.shape)
rf = RandomForestClassifier(n_estimators = 300, random_state = 42)
rf.fit(X_train, y_train)
predictions = rf.predict(X_test)

errors = abs(predictions - y_test)

#print('Mean Absolute Error:', round(np.mean(errors), 2))

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / y_test)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt

y_pred = [x[1] for x in rf.predict_proba(X_test)]
fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label = 1)

roc_auc = auc(fpr, tpr)

plt.figure(1, figsize = (15, 10))
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Checking roc_curve of the random forest model ')
plt.legend(loc="lower right")
plt.show()
import xgboost as xgb
X, y = X_tfidf,df_text['review_score']
data_dmatrix = xgb.DMatrix(data=X,label=y)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=0.2, random_state=123)
xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 10)
xg_reg.fit(X_train2,y_train2)

preds = xg_reg.predict(X_test2)
params = {"objective":"reg:linear",'colsample_bytree': 0.3,'learning_rate': 0.1,
                'max_depth': 5, 'alpha': 10}
import matplotlib.pyplot as plt

xgb.plot_tree(xg_reg,num_trees=0)
plt.rcParams['figure.figsize'] = [50, 10]
plt.show()
# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / y_test2)

# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt

y_pred = [x[1] for x in rf.predict_proba(X_test2)]
fpr, tpr, thresholds = roc_curve(y_test2, y_pred, pos_label = 1)

roc_auc = auc(fpr, tpr)

plt.figure(1, figsize = (15, 10))
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Checking roc_curve of the XGBoost model ')
plt.legend(loc="lower right")
plt.show()
# For you to install new packages, internet conectivity must be enabled
!pip install pycaret
from pycaret.nlp import *
df_text.dtypes
py_caret = setup(data = df_text, target = "review_comment_title", session_id = None)
lda = create_model('lda', multi_core = True)
lda_data = assign_model(lda)
lda_data.head()
from pycaret.classification import *
model = setup(data = lda_data, target = 'review_score',ignore_features=['review_comment_title','Dominant_Topic'])
compare_models()
