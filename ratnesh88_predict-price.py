import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/train.tsv',sep='\t')
data.head(15)
data.info()
data.isna().sum()
test = pd.read_csv('../input/test_stg2.tsv',sep='\t')
test.info()
for col in data.columns:
    if data[col].dtype ==  object:
        print(col)
        data[col].fillna('Missing',inplace=True)
        test[col].fillna('Missing',inplace=True)
        data[col] = data[col].str.strip().str.lower()
        test[col] = test[col].str.strip().str.lower()
        if col != 'item_description':
            ind = data[col].value_counts().index
            data[col] = pd.Series(data[col], dtype="category").cat.rename_categories(range(len(ind)))
            data[col] = data[col].astype(np.float32)
            ind = test[col].value_counts().index
            test[col] = pd.Series(test[col], dtype="category").cat.rename_categories(range(len(ind)))
            test[col] = test[col].astype(np.float32)
            
data.dtypes    
#ind = data['category_name'].value_counts().index
#data['category_name'] = pd.Series(data['category_name'], dtype="category").cat.rename_categories(range(len(ind)))
data.head()
features = ['name', 'item_condition_id', 'category_name', 'brand_name','shipping']
y_label = 'price'
# Reduce logging output.
tf.logging.set_verbosity(tf.logging.INFO)
for c in features[:-1]:
    n = max(data[c])//2
    data[c] = (data[c] - n )/n
    test[c] = (test[c] - n )/n
data.head()
#embedded_text_feature_column = hub.text_embedding_column(
#    key="item_description", 
 #   module_spec="https://tfhub.dev/google/nnlm-en-dim128/1")
# Training input on the whole training set with no limit on training epochs.
train_input_fn = tf.estimator.inputs.pandas_input_fn(
    data[features], data["price"], num_epochs=None, shuffle=True,batch_size=128)

# Prediction on the whole training set.
predict_train_input_fn = tf.estimator.inputs.pandas_input_fn(
    data[features], data["price"], shuffle=False,batch_size=128)
# Prediction on the test set.
predict_test_input_fn = tf.estimator.inputs.pandas_input_fn(
    test[features], shuffle=False,batch_size=128)
test.dtypes
feature_columns = [ tf.feature_column.numeric_column(c) for c in features]
estimator = tf.estimator.DNNRegressor(
    hidden_units=[512,256,128],
    feature_columns= feature_columns,
    optimizer=tf.train.AdagradOptimizer(learning_rate=0.003))
estimator.train(input_fn=train_input_fn, steps=5000)
estimator.evaluate(input_fn=predict_train_input_fn)
preds = list(estimator.predict(input_fn=predict_train_input_fn))
print(len(preds))
for i,p in enumerate(list(preds)[:15]):
    print(round(p['predictions'][0],3),data['price'][i])
preds = list(estimator.predict(input_fn=predict_test_input_fn))
print(len(preds))
pred = pd.Series(map(lambda x: round(x['predictions'][0],3) ,preds))
out = pd.DataFrame({'test_id':test['test_id'],'price':pred})
out.head()
out.to_csv('submit1.csv',index=False)
sample_submission = pd.read_csv('../working/submit1.csv',sep=',')
sample_submission.info()