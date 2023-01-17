import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.offline as pyo, plotly.graph_objs as go

import missingno as msno

import time



from sklearn.preprocessing import scale

from sklearn.model_selection import StratifiedKFold, train_test_split

from sklearn.metrics import *

from scikitplot.plotters import plot_ks_statistic

from scikitplot.metrics import plot_ks_statistic, plot_roc

from matplotlib import rcParams



from xgboost.sklearn import XGBClassifier

from sklearn.svm import OneClassSVM

from sklearn.neighbors import LocalOutlierFactor

from sklearn.covariance import EllipticEnvelope

from sklearn.ensemble import IsolationForest
class Stacking(object):

    def __init__(self, data, important_cols=None, fl_models=None, fl_models_names=None):

        self.data = data

        self.important_cols = important_cols

        self.fl_models = fl_models

        self.fl_models_names = fl_models_names

        for name in self.fl_models_names:

            self.data[name] = 0

        

        self.fl_data, self.sl_data = self._stacking_split()

        self.sl_model = XGBClassifier()

    

    

    def _fl_fit(self, X_train, Y_train):        

        for model in self.fl_models:

            model.fit(X_train, Y_train)

    

    

    def _fl_predict(self, X_test):

        preds = []

        for i, model in enumerate(self.fl_models):

             preds.append(model.predict(X_test).tolist())

        return np.array(preds).T.tolist()

            

    

    def _build_fl_models(self):

        kf = StratifiedKFold(n_splits=5, shuffle=True)

        all_indices = []

        preds = []

        for index, (train_indices, test_indices) in enumerate(kf.split(self.fl_data[important_cols], self.fl_data['Fraud'])):

            train = self.fl_data.iloc[train_indices]

            test = self.fl_data.iloc[test_indices]

            all_indices += test_indices.tolist()

            

            self._fl_fit(train[self.important_cols].values, train['Fraud'].values)

            preds += self._fl_predict(test[important_cols].values)

            print("{}/5 of data ended training".format(index+1))

        preds = pd.DataFrame(preds, columns=self.fl_models_names)

        self.fl_data = self.fl_data.iloc[all_indices]

        for name in self.fl_models_names:

            self.fl_data[name] = preds[name].values



            

    def _sl_fit(self, X_train, Y_train):

        self.sl_model.fit(X_train, Y_train)

        

        

    def _sl_predict(self, X_test):

        return self.sl_model.predict(X_test)

    

    

    def _sl_predict_probs(self, X_test):

        return self.sl_model.predict_proba(X_test)

    

    

    def run_stacked(self):

        tik = time.time()

        self._build_fl_models()

        self._sl_fit(self.fl_data[important_cols+self.fl_models_names], self.fl_data['Fraud'])

        tok = time.time()

        

        minutes = int((tok-tik) / 60)

        hours = int(minutes / 60)

        minutes = minutes % 60



        print("The Training Process Took {}h {} min".format(hours, minutes))

        

        preds = pd.DataFrame(self._fl_predict(self.sl_data[important_cols].values), columns=self.fl_models_names)

        for name in self.fl_models_names:

            self.sl_data[name] = preds[name].values

            

        self.preds = self._sl_predict(self.sl_data[important_cols+self.fl_models_names])

        self.probs = self._sl_predict_probs(self.sl_data[important_cols+self.fl_models_names])



        

    def _stacking_split(self, split_at=0.7):

        frauds = self.data[self.data['Fraud'] == 1]

        non_frauds = self.data[self.data['Fraud'] == -1]

        fraud_pivot, non_frauds_pivot = int(len(frauds)*split_at), int(len(non_frauds)*split_at)



        fl_data = pd.concat([frauds.iloc[:fraud_pivot], non_frauds.iloc[:non_frauds_pivot]])

        sl_data = pd.concat([frauds.iloc[fraud_pivot:], non_frauds.iloc[non_frauds_pivot:]])

        

        return fl_data, sl_data
# Sets the figure size to Seaborn and Matplotlib plots

rcParams['figure.figsize'] = [15, 15]

# Allows Plotly to plot offline

pyo.init_notebook_mode(connected=True)
col_names = ['V'+str(index) for index in range(29)]
data = pd.read_csv("../input/creditcard.csv")
data.tail(3)
msno.matrix(data)
data = data.rename(index=str, columns={"Class": "Fraud", "Time": "V0", "Amount": "V29"})

for col in ["V0", "V29"]:

    data[col] = scale(data[col].values)

    

data["Fraud"] = data["Fraud"].apply(lambda x: -1 if x else 1)
percentage = round(len(data[data.Fraud == -1]) / len(data) * 100, 4)

print("Percentage of Frauds is {}%".format(percentage))
col_names
X, Y = data[col_names].values, data["Fraud"].values



model = XGBClassifier()

model.fit(X, Y)

feat_importances = dict(zip(col_names, model.feature_importances_))
important_cols = []

importance = 0

for key, value in feat_importances.items():

    if(value > 0.015):

        important_cols.append(key)

        importance += value
print("Num. Features Initially: {}".format(len(feat_importances)))

print("Num. Features After: {}".format(len(important_cols)))

print("Preserved Importance: {}".format(round(importance, 4)))
figure, axes = plt.subplots(nrows=4, ncols=4, sharey=True)



for i in range(4):

    for j in range(4):

        if(i == 3 and j > 0):

            break

            

        axes[i, j].hist(data[important_cols[i*4 + j]].values, bins=75)

        axes[i, j].set_title(important_cols[i*4 + j])

        

plt.show()
sns.heatmap(data[important_cols].corr().values, 

            linewidth=0.5, 

            xticklabels=important_cols, 

            yticklabels=important_cols)



plt.title("Important Features Correlation Matrix")

plt.show()
models_names = ["EllipticEnvelope", 

                    "IsolationForest",

                    "LocalOutlierFactor"]



models = [EllipticEnvelope(support_fraction=0.7), 

              IsolationForest(behaviour="new", contamination="auto"), 

              LocalOutlierFactor(novelty=True, contamination="auto")]
%%capture

stack = Stacking(data=data[important_cols + ['Fraud']], 

                 important_cols=important_cols, 

                 fl_models=models, 

                 fl_models_names=models_names)
stack.run_stacked()
recs = []

precs = []

f1s = []



for name in models_names:

    prediction = stack.fl_data[name].values

    Y_fl = stack.fl_data['Fraud'].values

    precs.append(precision_score(Y_fl, prediction))

    recs.append(recall_score(Y_fl, prediction))

    f1s.append(f1_score(Y_fl, prediction))
trace1 = go.Bar(x=models_names, 

                y=precs, 

                name="Precision")

trace2 = go.Bar(x=models_names, 

                y=recs, 

                name="Recall")

trace3 = go.Bar(x=models_names, 

                y=f1s, 

                name="F1")



traces_list = [trace1, trace2, trace3]

layout = go.Layout(title="First Layer Models Performance")

figure = go.Figure(traces_list, layout)



pyo.iplot(figure)
%%capture



# Test the discordance rate over first level outputs

df_tmp = stack.fl_data[["Fraud"]]

df_tmp["outputs_sum"] = stack.fl_data[models_names].sum(axis=1).values



all_anomalies = len(df_tmp[df_tmp["Fraud"] == -1])

outputs_sum = df_tmp[df_tmp["Fraud"] == -1]["outputs_sum"].values



# If it is equals 3, all first level models agreed that it was 

# not an anomaly and consequently, if it is equals -3, all first 

# level models considered a record as an anomaly

outputs_sum = [0 if value == 3 or value == -3 else 1 for value in outputs_sum]
print(outputs_sum[:30])
print("The discordance among first level models is: {} (regarding anomalous records)"\

                                                  .format(sum(outputs_sum) / all_anomalies))
X = ['Precision','Recall','F1-Score']



Y_sl = stack.sl_data['Fraud']

Y = [np.round(precision_score(Y_sl, stack.preds)*100, 2), 

     np.round(recall_score(Y_sl, stack.preds)*100, 2),

     np.round(f1_score(Y_sl, stack.preds)*100, 2)]
trace = go.Bar(x=X, y=Y)

traces = [trace]



layout = go.Layout(title='Stack Performance Over Metrics')

figure = go.Figure(traces, layout)



pyo.iplot(figure)
# Sets the figure size to Seaborn and Matplotlib plots

rcParams['figure.figsize'] = [9, 9]
plot_roc(Y_sl, stack.probs, plot_micro=False, plot_macro=False)

plt.show()
plot_ks_statistic(Y_sl, stack.probs)

plt.show()