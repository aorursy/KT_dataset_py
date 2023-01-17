from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense, Flatten, Concatenate, concatenate, Dropout, Lambda
from keras.models import Model
from keras.layers.embeddings import Embedding
from tqdm import tqdm
import shap

# print the JS visualization code to the notebook
shap.initjs()
import pandas as pd
df = pd.read_csv('../input/adult.csv')
df.head()
df.dtypes
X_display = df.drop('income',axis=1)
y_display = df['income']
int_columns = df.select_dtypes(['int64']).columns
df[int_columns] = df[int_columns].astype('float32')

cat_columns = df.select_dtypes(['object']).columns
df[cat_columns] = df[cat_columns].astype('category')
df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)
X = df.drop('income',axis=1)
y = df['income']
#X,y = shap.datasets.adult()
#X_display,y_display = shap.datasets.adult(display=True)

# normalize data (this is important for model convergence)
dtypes = list(zip(X.dtypes.index, map(str, X.dtypes)))
for k,dtype in dtypes:
    if dtype == "float32":
        X[k] -= X[k].mean()
        X[k] /= X[k].std()

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=7)
X.head()
# build model
input_els = []
encoded_els = []
for k,dtype in dtypes:
    input_els.append(Input(shape=(1,)))
    if dtype == "int8":
        e = Flatten()(Embedding(X_train[k].max()+1, 1)(input_els[-1]))
    else:
        e = input_els[-1]
    encoded_els.append(e)
encoded_els = concatenate(encoded_els)
layer1 = Dropout(0.5)(Dense(100, activation="relu")(encoded_els))
out = Dense(1)(layer1)

# train model
regression = Model(inputs=input_els, outputs=[out])
regression.compile(optimizer="adam", loss='binary_crossentropy')
regression.fit(
    [X_train[k].values for k,t in dtypes],
    y_train,
    epochs=50,
    batch_size=512,
    shuffle=True,
    validation_data=([X_valid[k].values for k,t in dtypes], y_valid)
)
def f(X):
    return regression.predict([X[:,i] for i in range(X.shape[1])]).flatten()
explainer = shap.KernelExplainer(f, X.iloc[:100,:])
shap_values = explainer.shap_values(X.iloc[350,:], nsamples=500)
shap.force_plot(shap_values, X_display.iloc[350,:])
shap_values = explainer.shap_values(X.iloc[167,:], nsamples=500)
shap.force_plot(shap_values, X_display.iloc[167,:])
shap_values = explainer.shap_values(X.iloc[100:330,:], nsamples=500)
shap.force_plot(shap_values, X_display.iloc[100:330,:])
shap.summary_plot(shap_values50, X.iloc[100:330,:])
shap.dependence_plot("marital.status", 
                     shap_values, 
                     X.iloc[100:330,:], 
                     display_features=X_display.iloc[100:330,:])
