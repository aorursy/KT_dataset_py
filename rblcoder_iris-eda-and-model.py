
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
import cufflinks as cf
cf.set_config_file(offline=True)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from xgboost import XGBClassifier
from vecstack import stacking
df = pd.read_csv("../input/Iris.csv")
df.sample(5)
import seaborn as sns
import matplotlib.pyplot as plt


!pip install sweetviz
import sweetviz as sv
iris_report = sv.analyze(df)
iris_report.show_html('iris.html')
!ls
!conda install dtale -c conda-forge -y
!pip install flask_ngrok
#https://www.youtube.com/watch?v=8Il-2HHs2Mg
import pandas as pd

import dtale
import dtale.app as dtale_app

dtale_app.USE_NGROK = True
d=dtale.show(df)
d.main_url()
dtale.instances()
dtale.get_instance(1).kill()
#https://github.com/santosjorge/cufflinks/issues/185
!pip install plotly
!pip install cufflinks
df.info()
df.groupby(by='Species').describe().T
# https://seaborn.pydata.org/examples/scatterplot_matrix.html
ax = sns.pairplot(df, hue="Species")
# https://seaborn.pydata.org/generated/seaborn.boxplot.html
_ = sns.boxplot(x="Species", y="PetalLengthCm", data=df)
# https://seaborn.pydata.org/generated/seaborn.boxplot.html
_ = sns.boxplot(x="Species", y="PetalWidthCm", data=df)
# https://seaborn.pydata.org/generated/seaborn.boxplot.html
_ = sns.boxplot(x="Species", y="SepalLengthCm", data=df)
# https://seaborn.pydata.org/generated/seaborn.boxplot.html
_ = sns.boxplot(x="Species", y="SepalWidthCm", data=df)
# https://seaborn.pydata.org/tutorial/axis_grids.html
g = sns.FacetGrid(df, col="Species", hue="Species")
_=g.map(sns.kdeplot, "PetalLengthCm", "PetalWidthCm", alpha=.7)
_=g.add_legend()
# https://seaborn.pydata.org/tutorial/axis_grids.html
g = sns.FacetGrid(df, col="Species", hue="Species")
_=g.map(sns.kdeplot, "SepalLengthCm", "SepalWidthCm", alpha=.7)
_=g.add_legend()
# https://seaborn.pydata.org/tutorial/axis_grids.html
g = sns.FacetGrid(df, col="Species", hue="Species")
_=g.map(sns.kdeplot, "PetalLengthCm", "SepalWidthCm", alpha=.7)
_=g.add_legend()
# https://seaborn.pydata.org/tutorial/axis_grids.html
g = sns.FacetGrid(df, col="Species", hue="Species")
_=g.map(sns.kdeplot, "PetalWidthCm", "SepalLengthCm", alpha=.7)
_=g.add_legend()
df.Species.value_counts()
df[df.columns[1:5]].iplot(kind='hist')
for s in df.Species.unique():
    df.loc[df.Species==s, df.columns[1:5]].iplot(kind='hist', title=s)
df.Species.value_counts().iplot(kind='bar')
df[df.columns[1:5]].scatter_matrix()
df[df.columns[1:5]].iplot(kind='box')
for s in df.Species.unique():
    df.loc[df.Species==s, df.columns[1:5]].iplot(kind='box', title=s)
#df['colors'] = df['Species']
#https://stackoverflow.com/questions/21131707/multiple-data-in-scatter-matrix?rq=1
#df.colors.replace({'Iris-versicolor' : '#0392cf', 'Iris-virginica' : '#7bc043', 'Iris-setosa' : '#ee4035' }, inplace=True)
# color_wheel = {'Iris-versicolor': "#0392cf", 
#                'Iris-virginica': "#7bc043", 
#                'Iris-setosa': "#ee4035"}
# colors = df["Species"].map(lambda x: color_wheel.get(x))
#https://stackoverflow.com/questions/22943894/class-labels-in-pandas-scattermatrix
#df[df.columns[1:5]].scatter_matrix(color=colors)
df.Species.replace({'Iris-versicolor' : 3, 'Iris-virginica' : 2, 'Iris-setosa' : 1 }, inplace=True)
df.head()
y = df[['Species']]
X = df.loc[:,df.columns[1:5]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
clf = RandomForestClassifier(max_depth=4, n_estimators=100, random_state=0)
clf.fit(X_train, np.ravel(y_train))
y_pred = clf.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
X.columns
from sklearn.inspection import plot_partial_dependence
features = [2, 3, (0, 3), (1,2)]
plot_partial_dependence(clf, X, features, target=1, n_cols=4) 
# fig.set_figwidth(8)
# fig.set_figheight(15)
# fig.tight_layout()
plt.gcf().set_figwidth(8)
from sklearn.inspection import plot_partial_dependence
features = [2, 3, (0, 3), (1,2)]
plot_partial_dependence(clf, X, features, target=2, n_cols=4) 
# fig.set_figwidth(8)
# fig.set_figheight(15)
# fig.tight_layout()
plt.gcf().set_figwidth(8)
from sklearn.inspection import plot_partial_dependence
features = [2, 3, (0, 3), (1,2)]
plot_partial_dependence(clf, X, features, target=3, n_cols=4) 
# fig.set_figwidth(8)
# fig.set_figheight(15)
# fig.tight_layout()
plt.gcf().set_figwidth(8)
clf.classes_
y_test[:5]
import shap

# load JS visualization code to notebook
shap.initjs()

explainer = shap.KernelExplainer(clf.predict_proba, X_train)
shap_values = explainer.shap_values(X_test)


# plot the SHAP values for the Setosa output of the first instance
shap.force_plot(explainer.expected_value[0], shap_values[0][0,:], X_test.iloc[0,:])
shap.force_plot(explainer.expected_value[1], shap_values[1][0,:], X_test.iloc[0,:])
shap.force_plot(explainer.expected_value[2], shap_values[2][0,:], X_test.iloc[0,:])
shap_values = explainer.shap_values(X_test)

shap.force_plot(explainer.expected_value[0], shap_values[0], X_test)
shap.force_plot(explainer.expected_value[1], shap_values[1], X_test)
shap.force_plot(explainer.expected_value[2], shap_values[2], X_test)
shap.dependence_plot("PetalLengthCm", shap_values[0], X_test)
shap.dependence_plot("PetalWidthCm", shap_values[0], X_test)
shap.dependence_plot("PetalLengthCm", shap_values[1], X_test)
shap.dependence_plot("PetalWidthCm", shap_values[1], X_test)
shap.dependence_plot("PetalLengthCm", shap_values[2], X_test)
shap.dependence_plot("PetalWidthCm", shap_values[2], X_test)
shap.summary_plot(shap_values[0], X_test)
shap.summary_plot(shap_values[1], X_test)
shap.summary_plot(shap_values[2], X_test)
shap.summary_plot(shap_values, X_test, plot_type="bar")
sgd_clf = SGDClassifier(random_state=0)
sgd_clf.fit(X_train, np.ravel(y_train))
y_pred = sgd_clf.predict(X_test)
print(classification_report(y_test, y_pred))
log_clf = LogisticRegression(multi_class='ovr', solver='lbfgs')
log_clf.fit(X_train, np.ravel(y_train))
y_pred = log_clf.predict(X_test)
print(classification_report(y_test, y_pred))
xgb_clf = XGBClassifier(random_state=0, n_jobs=-1, learning_rate=0.1, 
                   n_estimators=100, max_depth=3)
xgb_clf.fit(X_train, np.ravel(y_train))
y_pred = xgb_clf.predict(X_test)
print(classification_report(y_test, y_pred))
models = [
#     KNeighborsClassifier(n_neighbors=5,
#                         n_jobs=-1),
    SGDClassifier(random_state=0),
        
#     RandomForestClassifier(random_state=0, n_jobs=-1, 
#                            n_estimators=100, max_depth=3),
     RandomForestClassifier(random_state=0),    
#     XGBClassifier(random_state=0, n_jobs=-1, learning_rate=0.1, 
#                   n_estimators=100, max_depth=3)
    LogisticRegression(random_state=0,multi_class='ovr', solver='lbfgs')
]

S_train, S_test = stacking(models,                   
                           X_train, np.ravel(y_train), X_test,   
                           regression=False, 
     
                           mode='oof_pred_bag', 
       
                           needs_proba=False,
         
                           save_dir=None, 
            
                           metric=accuracy_score, 
    
                           n_folds=4, 
                 
                           stratified=True,
            
                           shuffle=True,  
            
                           random_state=0,    
         
                           verbose=2)
S_train
S_train.shape
S_test
S_test.shape
# model = XGBClassifier(random_state=0, n_jobs=-1, learning_rate=0.1, 
#                       n_estimators=100, max_depth=3)
model = LogisticRegression(multi_class='ovr', solver='lbfgs')    
model = model.fit(S_train, np.ravel(y_train))
y_pred = model.predict(S_test)
print('Final prediction score: [%.8f]' % accuracy_score(y_test.values, y_pred))
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))