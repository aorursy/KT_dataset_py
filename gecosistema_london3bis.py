

import numpy as np

import plotly

import pandas as pd

import pylab

import matplotlib.pyplot as plt

import calendar

import seaborn

import math



from sklearn.svm import SVR

from sklearn.cross_validation import train_test_split

from sklearn.grid_search import GridSearchCV, RandomizedSearchCV

from sklearn import preprocessing

from sklearn.metrics import r2_score, mean_squared_error

from scipy.stats import randint as sp_randint

from scipy.stats import uniform as sp_uniform

from scipy import stats

from sklearn.model_selection import cross_val_score

from math import sqrt



%matplotlib inline
workdir='../input/'
flowrate_data=pd.read_csv(workdir+"3bislondon.csv")
flowrate_data.head()
flowrate_data.describe()
import matplotlib.pyplot as plt

import matplotlib.mlab as mlab

import numpy as np



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

#py.sign_in('valluzzi', 'rlfa21c6rl')



n = 50

x, y, z, s, ew = np.random.rand(5, n)

c, ec = np.random.rand(2, n, 4)

area_scale, width_scale = 500, 5



fig, ax = plt.subplots()

sc = ax.scatter(x, y, c=c,

                s=np.square(s)*area_scale,

                edgecolor=ec,

                linewidth=ew*width_scale)

ax.grid()



plot_url = py.plot_mpl(fig)

%matplotlib inline



#https://seaborn.pydata.org/tutorial/distributions.html



g = seaborn.pairplot(flowrate_data)
g = seaborn.pairplot(flowrate_data, diag_kind="kde")
g = seaborn.PairGrid(flowrate_data)

g.map(plt.scatter)






#https://seaborn.pydata.org/examples/many_pairwise_correlations.html



# Compute the correlation matrix

corr = flowrate_data.corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(20, 10))



# Generate a custom diverging colormap

cmap = seaborn.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

seaborn.heatmap(corr, mask=mask, cmap=cmap, center=0,annot=True,

            square=False, linewidths=1, cbar_kws={"shrink": .5})



plt.savefig('./correlation_matrix.png', dpi=300)
F1=flowrate_data.columns[1] # seleziona la colonna che vuoi analizzare

F1X=flowrate_data[F1]

min=np.min(F1X)

max=np.min(F1X)
stats.probplot(F1X, dist="norm", plot=pylab)



plt.savefig('./probability_plot.png', dpi=150)
features_cols=flowrate_data.columns[1:4]

target_col=flowrate_data.columns[0]

print ("Feature column(s):\n{}\n".format(features_cols))

print ("Target column:\n{}".format(target_col))



print (flowrate_data.head())

print (flowrate_data.describe())
target_col
features_cols
X=flowrate_data[features_cols] ## features

y=flowrate_data[target_col] ## target
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.25)



# If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split. 

#If int, represents the absolute number of test samples. 

#If None, the value is automatically set to the complement of the train size. 

#If train size is also None, test size is set to 0.25.



##X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)



print ("X_train counts="+str(np.shape(X_train)))

print ("X_test counts="+str(np.shape(X_test)))

print ("y_train counts="+str(np.shape(y_train)))

print ("y_test counts="+str(np.shape(y_test)))





# save to CSV

pd.DataFrame(y_train).to_csv('y_train_v1.1.csv')

pd.DataFrame(X_train).to_csv('X_train_v1.1.csv')

pd.DataFrame(X_test).to_csv('X_test_v1.1.csv')

pd.DataFrame(y_test).to_csv('y_test_v1.1.csv')

from sklearn.preprocessing import MinMaxScaler

mms = MinMaxScaler()

X_train_norm = mms.fit_transform(X_train)

X_test_norm = mms.transform(X_test)



from sklearn.preprocessing import StandardScaler



stdsc = StandardScaler()

X_train_std = stdsc.fit_transform(X_train)

X_test_std = stdsc.transform(X_test)



X_std=stdsc.transform(X)



#define index array

X_test_index=X_test.index

X_train_index=X_train.index

X_index=X.index



# save to CSV

np.savetxt('X_test_std.csv', np.column_stack((X_test_index, X_test_std)), fmt='%.4f', delimiter=',')

np.savetxt('X_train_std.csv', np.column_stack((X_train_index, X_train_std)), fmt='%.4f', delimiter=',')

np.savetxt('X_std.csv', np.column_stack((X_index, X_std)), fmt='%.4f', delimiter=',')









#pd.DataFrame(X_test_std).to_csv(workdir+'X_test_std.csv')

#pd.DataFrame(X_train_std).to_csv(workdir+'X_train_std.csv')

#pd.DataFrame(X_std).to_csv(workdir+'X_std.csv')





from sklearn.feature_selection import RFE

from sklearn.svm import SVR



feat_toselect=3 #define how many feature you want to be selected



# create a base classifier used to evaluate a subset of attributes

svr = SVR(kernel='linear')

# create the RFE model and select 3 attributes

rfe = RFE(svr, feat_toselect)  

rfe = rfe.fit(X_std, y)

# summarize the selection of the attributes

print(rfe.support_)

print(rfe.ranking_)
from sklearn.ensemble import ExtraTreesClassifier



from sklearn.feature_selection import SelectFromModel



clf = ExtraTreesClassifier()

yy=np.asarray(y,dtype="|S6")

clf = clf.fit(X_std, yy)

clf.feature_importances_  



model = SelectFromModel(clf, prefit=True)

X_std_new = model.transform(X_std)

X_std_new.shape

#print (X_std[1,1:])

#print (X_std_new[1,1:])
from sklearn.pipeline import Pipeline

from sklearn.svm import SVR

from sklearn.ensemble import RandomForestClassifier

clf = Pipeline([

  ('feature_selection', SelectFromModel(SVR(kernel="linear"))),

  ('classification', RandomForestClassifier())

])

clf.fit(X_std, yy)

clf.named_steps['feature_selection'].get_support()
from sklearn.ensemble import RandomForestClassifier

feat_labels = flowrate_data.columns[1:]

forest = RandomForestClassifier(n_estimators=200,

                                random_state=0,

                                n_jobs=-1)

yy_train=np.asarray(y_train,dtype="|S6")

forest.fit(X_train, yy_train)

importances = forest.feature_importances_



indices = np.argsort(importances)[::-1]



for f in range(X_train.shape[1]):

    print("%2d) %-*s %f" % (f + 1, 30, 

                            feat_labels[indices[f]], 

                            importances[indices[f]]))



plt.title('Feature Importances')

plt.bar(range(X_train.shape[1]), 

        importances[indices],

        color='lightblue', 

        align='center')



plt.xticks(range(X_train.shape[1]),

feat_labels[indices], rotation=90)



plt.xlim([-1, X_train.shape[1]])

plt.tight_layout()

plt.savefig('./random_forest.png', dpi=300)

plt.show()
X_train=X_train_std

X_test=X_test_std





#Training SVR

svr=SVR()

svr.fit(X_train, y_train)



print (y_test)
# Validation SVR



svr_pred = svr.predict(X_test)



score_svr = r2_score(y_test, svr_pred)

rmse_svr = sqrt(mean_squared_error(y_test, svr_pred))

from sklearn.metrics import mean_squared_error

mse_svr=mean_squared_error(y_test, svr_pred)





print("Score SVR: %f" % score_svr)

print("RMSE SVR: %f" % rmse_svr)

print("MSE SVR: %f" % mse_svr)
# Define Parameters



k=['rbf','linear']

c= range(1,2000,1) # from 1 to 2000 with steps=10

g=np.arange(1e-4,1e-1,0.0005)

g=g.tolist()

e=np.arange(1e-4,1e-1,0.0005) # from 0.001 to 0.1 with steps=0.005

e=e.tolist()



#c=np.logspace(0, 20, num=10, base=2) # logarithmic scale

#c=c.tolist()

#g=np.logspace(-20, 1, num=10, base=2) # logarithmic scale

#g=g.tolist()

#e=np.logspace(-20, 1, num=10, base=2) # logarithmic scale

#e=e.tolist()



param_dist=dict(kernel=k, C=c, gamma=g,epsilon=e)

#print (param_dist)
#param_dist = {  'C': sp_uniform (1000, 10000), 

#                'kernel': ['linear','rbf']

#             }



n_iter_search = 5



# MSE optimized

#SVR_tuned_RS = RandomizedSearchCV(SVR (C=1), param_distributions = param_dist, scoring = 'mean_squared_error', n_iter=n_iter_search)



# R^2 optimized

SVR_tuned_RS = RandomizedSearchCV(SVR (C=1), param_distributions = param_dist, scoring = 'r2', n_iter=n_iter_search, n_jobs=-1, cv=5)



# Fit

SVR_tuned_RS.fit(X_train, y_train)



# Best score and corresponding parameters.

print('best CV score from grid search: {0:f}'.format(SVR_tuned_RS.best_score_))

print('corresponding parameters: {}'.format(SVR_tuned_RS.best_params_))



# Predict and score

svr_tuned_pred_RS= SVR_tuned_RS.predict(X_test)



score_svr_tuned_RS = r2_score(y_test, svr_tuned_pred_RS)

rmse_svr_tuned_RS = sqrt(mean_squared_error(y_test, svr_tuned_pred_RS))



from sklearn.metrics import mean_squared_error

mse_svr_tuned_RS=mean_squared_error(y_test, svr_tuned_pred_RS)



print('SVR Results\n')



print("Score SVR: %f" % score_svr)

print("Score SVR tuned RS: %f" % score_svr_tuned_RS)



print("\nRMSE SVR: %f" % rmse_svr)

print("RMSE SVR tuned RS: %f" % rmse_svr_tuned_RS)



print("\nMSE SVR: %f" % mse_svr)

print("MSE SVR tuned RS: %f" % mse_svr_tuned_RS)



# Results

print("Results as RMSE")

print('\n')

print("SVR: %f" % rmse_svr)

print("SVR tuned RS: %f" % rmse_svr_tuned_RS)

print("SVR: %f" % mse_svr)

print("SVR tuned RS: %f" % mse_svr_tuned_RS)

print('\n')
print (SVR_tuned_RS.predict(X_test))
#visualization





predict_RS_test = SVR_tuned_RS.predict(X_test)

#predict_GS_test = svr_tuned_GS.predict(X_test)

predict_RS_train = SVR_tuned_RS.predict(X_train)

#predict_GS_train = svr_tuned_GS.predict(X_train)



#label array

test_labels=np.ones(len(predict_RS_test))

train_labels=np.ones(len(predict_RS_train))*2

# label=1 test label=2 train

#concatenate index and save

np.savetxt('predict_RS_test.csv', np.column_stack((X_test_index, predict_RS_test,y_test,test_labels)), fmt='%.4f', delimiter=',', header='id1,pred_rs,true,label')

#np.savetxt(workdir+'predict_GS_test.csv', np.column_stack((X_test_index, predict_GS_test,y_test,test_labels)), fmt='%.2f', delimiter=',', header='id1,pred_gs,true,label')

np.savetxt('predict_RS_train.csv', np.column_stack((X_train_index, predict_RS_train,y_train,train_labels)), fmt='%.4f', delimiter=',', header='id1,pred_rs,true,label')

#np.savetxt(workdir+'predict_GS_train.csv', np.column_stack((X_train_index, predict_GS_train,y_train,train_labels)), fmt='%.2f', delimiter=',', header='id1,pred_gs,true,label')



#append 

a=np.column_stack((X_test_index, predict_RS_test,y_test,test_labels))

b=np.column_stack((X_train_index, predict_RS_train,y_train,train_labels))



RS_append=np.append(a,b,axis=0)



#a=np.column_stack((X_test_index, predict_GS_test,y_test,test_labels))

#b=np.column_stack((X_train_index, predict_GS_train,y_train,train_labels))



#GS_append=np.append(a,b,axis=0)





np.savetxt('predict_RS.csv',RS_append, fmt='%.4f', delimiter=',', header='uid,pred_rs,true,label')

np.savetxt('predict_RS_excel_v1.1.csv',RS_append, fmt='%.4f', delimiter=';', header='uid;pred_rs;true;label')

#np.savetxt(workdir+'predict_GS.csv',GS_append, fmt='%.2f', delimiter=',', header='uid,pred_gs,true,label')





# read as pandas



predict_RS_pd=pd.read_csv('predict_RS.csv')

#predict_GS_pd=pd.read_csv(workdir+'predict_GS.csv')

#plot



import matplotlib.pyplot as plt



predict_RS_pd.sort_values(['# uid']).plot(x='# uid', y=(['pred_rs','true']))



plt.savefig('./pred_RS.png', dpi=150)

plt.show()
# Results

print("Results as RMSE")

print('\n')

print("SVR: %f" % rmse_svr)

#print("SVR tuned GS: %f" % rmse_svr_tuned_GS)

print("SVR tuned RS: %f" % rmse_svr_tuned_RS)

print('\n')
