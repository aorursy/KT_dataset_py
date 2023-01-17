import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import os, math

from datetime import datetime

import seaborn as sns

import matplotlib.cm as cm



from sklearn import neighbors, linear_model, svm, tree, ensemble

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import confusion_matrix,roc_auc_score

from sklearn import metrics

from sklearn.model_selection import train_test_split
dat = pd.read_csv("../input/ATP.csv")
dat['date'] = dat.tourney_date.apply(lambda t: datetime.strptime(str(t), '%Y%m%d'))
colnames = dict()

colnames['type1'] = ['1stIn', '1stWon', '2ndWon', 'SvGms', 'ace', 'bpFaced', 'bpSaved', 'df', 'svpt']

colnames['type2'] = ['age', 'entry', 'hand', 'ht', 'id', 'ioc', 'name', 'rank', 'rank_points', 'seed']

colnames['type3'] = ['best_of', 'draw_size', 'match_num', 'minutes', 'round', 'score', 'surface', 'tourney_date',

                     'tourney_id', 'tourney_level', 'tourney_name', 'date']
df = pd.DataFrame()

mat = []

for i in dat.index:

    row = []

    for col in colnames['type3']:

        row.append(dat[col][i])

    if i % 2 == 0: #j0

        # j0=loser, j1=winner

        for col in colnames['type1']:

            row.append(dat['l_'+col][i])

        for col in colnames['type2']:

            row.append(dat['loser_'+col][i])

        for col in colnames['type1']:

            row.append(dat['w_'+col][i])

        for col in colnames['type2']:

            row.append(dat['winner_'+col][i])

        row.append(1) #target winner --> j1

    else: #j1

        # j0=winner, j1=loser

        for col in colnames['type1']:

            row.append(dat['w_'+col][i])

        for col in colnames['type2']:

            row.append(dat['winner_'+col][i])

        for col in colnames['type1']:

            row.append(dat['l_'+col][i])

        for col in colnames['type2']:

            row.append(dat['loser_'+col][i])

        row.append(0) #target winner --> j0

    mat.append(row)
colDataFrame = colnames['type3']

for col in colnames['type1']:

    colDataFrame.append('j0_'+col)

for col in colnames['type2']:

    colDataFrame.append('j0_'+col)

for col in colnames['type1']:

    colDataFrame.append('j1_'+col)

for col in colnames['type2']:

    colDataFrame.append('j1_'+col)

colDataFrame.append("target")
df = pd.DataFrame(columns=colDataFrame, data=mat)
df.head()
print("nRows : {}, nCols : {}".format(df.shape[0], df.shape[1]))
dfe = df.copy()
dfe = dfe.loc[np.invert(dfe.j0_rank.isna()) & np.invert(dfe.j1_rank.isna())]

dfe = dfe.loc[np.invert(dfe.j0_rank_points.isna()) & np.invert(dfe.j1_rank_points.isna())]
dfe = dfe.loc[np.invert(dfe.surface.isna())]

dfe = dfe.loc[dfe.surface != "None"]
print("nRows : {}, nCols : {}".format(dfe.shape[0], dfe.shape[1]))
print("There are {} matches.".format(dfe.shape[0]))

print("There are {} different players.".format(len(list(set(dfe.j0_name + dfe.j1_name)))))
# pie chart of surface

count_surface = dfe[["tourney_id", "surface"]]

count_surface = count_surface.groupby(["surface"]).agg('count')

count_surface.reset_index(inplace=True)

count_surface.columns=["surface","Count"]

count_surface.sort_values("Count", inplace=True)



x = np.arange(count_surface.shape[0])

ys = [i+x+(i*x)**2 for i in range(count_surface.shape[0])]

colors = cm.rainbow(np.linspace(0, 1, len(ys)))



plt.rc('font', weight='bold')

f, ax = plt.subplots(figsize=(5,5), dpi=120)

labels=count_surface.surface.values



sizes=count_surface.Count.values



explode = [0.9 if sizes[i] < 1000 else 0.0 for i in range(len(sizes))]

ax.pie(sizes, explode = explode, labels=labels, colors = colors,

       autopct = lambda x:'{:1.0f}%'.format(x) if x > 1 else '',

       shadow = False, startangle=0, textprops={'fontsize': 7})

ax.axis('equal')

ax.set_title('Surface', bbox={'facecolor':'blue', 'pad':3}, color = 'w', fontsize=10)

_ = ax
# bpFaced barplot

dfe_bpFaced = dfe.groupby(["target"])["j0_bpFaced","j1_bpFaced"].agg('mean')

dfe_bpFaced.reset_index(inplace=True)



players=['player 0', 'player 1']

lw=[0,1]

pos = np.arange(len(players))

bar_width = 0.35

index_loser=dfe_bpFaced.iloc[0,1:].values

index_winner=dfe_bpFaced.iloc[1,1:].values



plt.bar(pos,index_loser,bar_width,color='red',edgecolor='black')

plt.bar(pos+bar_width,index_winner,bar_width,color='green',edgecolor='black')

plt.xticks(pos+0.1, players)

plt.xlabel('', fontsize=16)

plt.ylabel('Breakpoints faced', fontsize=15)

plt.title('Barchart',fontsize=18)

plt.legend(lw,loc=2)

plt.show()
# Binarize surface

df_surface = dfe.surface.str.get_dummies()

df_surface.head()
ax = sns.boxplot(x="surface", y="j1_rank_points", hue="target", data=dfe, palette=['blue','red'])

ax.set_ylim([0, 5000])

ax.set_ylabel("rank points")

_ = ax
ax = sns.boxplot(x="target", y="j1_rank_points", data=dfe, palette=["blue","red"])

ax.set_ylim([0,4000])

ax.set_ylabel('rank points')

_ = ax
class Model:

    def __init__(self,data,seed,random_sample):

        self.random_sample = random_sample

        self.seed = seed

        

        self.data = data.sample(frac=self.random_sample, replace=False, random_state=self.seed)

        

        self.lr=None

        self.pred_train=None

        self.pred_test=None

    def split(self, test_size):

        self.test_size = test_size

        train_X, test_X, train_y, test_y = train_test_split(self.data,self.data['target'], test_size = test_size, random_state=self.seed)

        self.train_X = train_X.drop(columns=['target'])

        self.test_X = test_X.drop(columns=['target'])

        self.train_y = train_y

        self.test_y = test_y

    def model_LR(self,n_jobs,cv,regul):

        self.regul = regul

        if regul=='none':

            n_iters = np.array([50, 200])

            model = linear_model.SGDClassifier(loss='log', random_state=0, penalty=self.regul)

            grid = GridSearchCV(estimator=model, param_grid=dict(n_iter_no_change=n_iters), scoring='roc_auc', n_jobs=n_jobs, cv=cv, verbose=1)

            grid.fit(self.train_X,self.train_y)

            self.grid = grid

        elif regul=='elasticnet':

            n_iters = np.array([50, 200])

            alphas = np.logspace(-5, 1, 5)

            l1_ratios = np.array([0, 0.15, 0.3, 0.4, 0.5, 0.6, 0.85, 1])

            model = linear_model.SGDClassifier(loss='log', random_state=0, penalty=self.regul,n_iter_no_change=100,max_iter=100)

            grid = GridSearchCV(estimator=model, param_grid=dict(alpha=alphas,l1_ratio=l1_ratios), scoring='roc_auc', n_jobs=n_jobs, cv=cv, verbose=1)

            grid.fit(self.train_X,self.train_y)

            self.grid = grid

        return self.lr

    def model_GB(self,n_jobs,cv):

        param_grid = {'n_estimators' : [10, 20, 30, 40, 50, 60, 70, 80, 90, 100,150,200],

                      'learning_rate':[0.1,0.2,0.5,0.7,0.9,1]}

        model = ensemble.GradientBoostingClassifier()

        grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='roc_auc', n_jobs=n_jobs, cv=cv, verbose=1)

        grid.fit(self.train_X,self.train_y)

        self.grid = grid

    def model_KNN(self,n_jobs,cv):

        param_grid = {'n_neighbors': np.arange(1,310,10)}

        model = neighbors.KNeighborsClassifier()

        grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='roc_auc', n_jobs=n_jobs, cv=cv, verbose=1)

        grid.fit(self.train_X,self.train_y)

        self.grid = grid

    def model_RF(self,n_jobs,cv):

        param_grid = {'criterion' : ['entropy', 'gini'],

                      'n_estimators' : [20, 40, 60, 80, 100, 120, 160, 200, 250, 300],

                      'max_features' :['sqrt', 'log2']}

        model = ensemble.RandomForestClassifier()

        grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='roc_auc', n_jobs=n_jobs, cv=cv, verbose=1)

        grid.fit(self.train_X,self.train_y)

        self.grid = grid

    def predict(self):

        self.pred_train = self.grid.best_estimator_.predict_proba(X=self.train_X)

        self.pred_test = self.grid.best_estimator_.predict_proba(X=self.test_X)

    def get_AUC(self):

        self.train_auc=metrics.roc_auc_score(y_score=self.grid.best_estimator_.predict_proba(X=self.train_X)[:,1], y_true=self.train_y)

        self.test_auc=metrics.roc_auc_score(y_score=self.grid.best_estimator_.predict_proba(X=self.test_X)[:,1], y_true=self.test_y)

        return (self.train_auc,self.test_auc)

    ### get contingency table + recall precision + roc curve !!!

    def boxplot(self):

        plt.figure()

        plt.subplot(1,2,1)

        sns.boxplot(x=self.train_y.values, y=self.grid.best_estimator_.predict_proba(X=self.train_X.values)[:,1])

        plt.title('Train')

        plt.subplot(1,2,2)

        sns.boxplot(x=self.test_y.values, y=self.grid.best_estimator_.predict_proba(X=self.test_X.values)[:,1])

        plt.title('Test')

        return plt

    def rocCurve(self):

        plt.figure()

        plt.subplot(1,2,1)

        fpr, tpr, thresholds = metrics.roc_curve(y_score=self.grid.best_estimator_.predict_proba(X=self.train_X)[:,1], y_true=self.train_y)

        plt.plot(fpr, tpr,'r')

        plt.plot([0,1],[0,1],'b')

        plt.title('Train, AUC: {}'.format(round(metrics.auc(fpr,tpr),3)))

        

        plt.subplot(1,2,2)

        fpr, tpr, thresholds = metrics.roc_curve(y_score=self.grid.best_estimator_.predict_proba(X=self.test_X)[:,1], y_true=self.test_y)

        plt.plot(fpr, tpr,'r')

        plt.plot([0,1],[0,1],'b')

        plt.title('Test, AUC: {}'.format(round(metrics.auc(fpr,tpr),3)))

        return plt

    def confusion(self,set_):

        if set_ == "train":

            res = metrics.confusion_matrix(y_true=self.train_y,y_pred=self.pred_train)

        elif set_ == "test":

            res = metrics.confusion_matrix(y_true=self.test_y,y_pred=self.pred_test)

        return res

    def getAccuracy(self):

        res=(metrics.accuracy_score(y_true=self.train_y,y_pred=self.pred_train),

            metrics.accuracy_score(y_true=self.test_y,y_pred=self.pred_test))

        return res

    def getClassificationReport(self,set_):

        if set_ == "train":

            res = metrics.classification_report(self.train_y, self.pred_train)

        elif set_ == "test":

            res = metrics.classification_report(self.test_y, self.pred_test)

        return res
dfm = dfe[["target","j0_rank_points","j1_rank_points","j0_bpFaced","j1_bpFaced"]]

dfm[df_surface.columns] = df_surface

dfm.dropna(inplace=True)



dfm.j0_rank_points = dfm.j0_rank_points.apply(lambda x: math.log(x))

dfm.j1_rank_points = dfm.j1_rank_points.apply(lambda x: math.log(x))
dfm.head()
lr = Model(data=dfm,seed=123,random_sample=1)

lr.split(0.35)

lr.model_LR(cv=4,n_jobs=8,regul="elasticnet")

lr.predict()
_ = lr.rocCurve()
_ = lr.boxplot()
lr.grid.best_params_
gb = Model(data=dfm,seed=123,random_sample=1)

gb.split(0.35)

gb.model_GB(cv=4,n_jobs=8)

gb.predict()
_ = gb.rocCurve()
_ = gb.boxplot()
gb.grid.best_params_
knn = Model(data=dfm,seed=123,random_sample=1)

knn.split(0.35)

knn.model_KNN(cv=4,n_jobs=8)

knn.predict()
_ = knn.rocCurve()
_ = knn.boxplot()
knn.grid.best_params_
rf = Model(data=dfm,seed=123,random_sample=1)

rf.split(0.35)

rf.model_RF(cv=4,n_jobs=4)

rf.predict()
_ = rf.rocCurve()
_ = rf.boxplot()
rf.grid.best_params_