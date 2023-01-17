import numpy as np

import pandas as pd

from sklearn.linear_model import LogisticRegression

from sklearn import tree

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import log_loss

from graphviz import Source

import matplotlib.pyplot as plt
# FIRST DATA SET

X1=np.array([[1],[2],[3],[4],[5]])

y1=np.array([[0],[0],[1],[0],[0]]) # with bad y=1 at x=3, we will get depth=2



# REORDERED

X2=np.array([[0],[1],[2],[3],[4]])

y2=np.array([[0],[0],[0],[0],[1]]) # nice x -> y, we will get depth =1



# TEST ONE HOT ENCODE

X3=np.array([[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,0,1],[0,0,0,0,1]])

y3=np.array([[0],[0],[1],[0],[0]])



# TEST ONE HOT ENCODE (x1=1 and x3=1)

X4=np.array([[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,0,1],[0,0,0,0,1]])

y4=np.array([[1],[0],[1],[0],[0]])



# TEST LINEAR MODELS:

X5=np.array([[0],[1],[2],[3],[10000000]])  #nice X values, it's good to linear models

y5=np.array([[0],[0],[0],[0],[1]]) # nice x->y
#http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html

#    I will set some parameters, the main parameter is class_weight

#

#    criterion

#       The function to measure the quality of a split. 

#       Supported criteria are “gini” for the Gini impurity and 

#       “entropy” for the information gain. <- i like this, but let's use gini...

#

#    class_weight

#        The “balanced” mode uses the values of y to automatically adjust weights 

#        inversely proportional to class frequencies in the input data as 

#

#        weight= n_samples / (n_classes * np.bincount(y))

#        

#    max_depth

#        The maximum depth of the tree. If None, then nodes are expanded until all 

#        leaves are pure or until all leaves contain less than min_samples_split samples.



model=DecisionTreeClassifier(criterion='gini',class_weight='balanced',max_depth=None)
# dataset 1

model.fit(X1,y1)

y_hat1=model.predict_proba(X1)[:,1]

loss1 =log_loss(y1,y_hat1)

print("dataset1: ")

print('    depth:  ',model.tree_.max_depth)

print('    proba:  ',y_hat1)

print('    logloss:',loss1)

#plot tree :)

Source( tree.export_graphviz(model, out_file=None))
# dataset 2

model.fit(X2,y2)

y_hat2=model.predict_proba(X2)[:,1]

loss2 =log_loss(y2,y_hat2)

print("dataset2: ")

print('    depth:  ',model.tree_.max_depth)

print('    proba:  ',y_hat1)

print('    logloss:',loss1)

#plot tree :)

Source( tree.export_graphviz(model, out_file=None))
# dataset 5

model.fit(X5,y5)

y_hat5=model.predict_proba(X5)[:,1]

loss5 =log_loss(y5,y_hat5)

print("dataset5: ")

print('    depth:  ',model.tree_.max_depth)

print('    proba:  ',y_hat5)

print('    logloss:',loss5)

#plot tree :)

Source( tree.export_graphviz(model, out_file=None))
# dataset 3

print('X=',X3)

print('Y=',y3)

model.fit(X3,y3)

y_hat3=model.predict_proba(X3)[:,1]

loss3 =log_loss(y3,y_hat3)

print("dataset3: ")

print('    depth:  ',model.tree_.max_depth)

print('    proba:  ',y_hat3)

print('    logloss:',loss3)

#plot tree :)

Source( tree.export_graphviz(model, out_file=None))

# dataset 4

print('X=',X4)

print('Y=',y4)

model.fit(X4,y4)

y_hat4=model.predict_proba(X4)[:,1]

loss4 =log_loss(y4,y_hat4)

print("dataset3: ")

print('    depth:  ',model.tree_.max_depth)

print('    proba:  ',y_hat3)

print('    logloss:',loss3)

#plot tree :)

Source( tree.export_graphviz(model, out_file=None))

#Y values

plt.title('Dataset 1')

plt.plot(y1,label='Y_true')

plt.plot(y_hat1,label='y_hat1 - log loss:'+str(loss1))

plt.legend()

plt.show()



plt.title('Dataset 2')

plt.plot(y2,label='Y_true')

plt.plot(y_hat2,label='y_hat2 - log loss:'+str(loss2))

plt.legend()

plt.show()



plt.title('Dataset 3')

plt.plot(y3,label='Y_true')

plt.plot(y_hat3,label='y_hat3 - log loss:'+str(loss3))

plt.legend()

plt.show()



plt.title('Dataset 4')

plt.plot(y4,label='Y_true')

plt.plot(y_hat4,label='y_hat4 - log loss:'+str(loss4))

plt.legend()

plt.show()



plt.title('Dataset 5')

plt.plot(y5,label='Y_true')

plt.plot(y_hat5,label='y_hat5 - log loss:'+str(loss5))

plt.legend()

plt.show()
#http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

model=LogisticRegression()



# try changing penalty and C hyperparameters :)
model.fit(X1,y1.ravel())

y_hat1=model.predict_proba(X1)[:,1]

loss1 =log_loss(y1,model.predict_proba(X1)[:,1])

print("model1: ")

print('    coefs:  ',model.coef_)

print('    y_true: ',y1.ravel())

print('    proba:  ',y_hat1)

print('    logloss:',loss1)

model.fit(X2,y2.ravel())

y_hat2=model.predict_proba(X2)[:,1]

loss2 =log_loss(y2,model.predict_proba(X2)[:,1])

print("model2: ")

print('    coefs:  ',model.coef_)

print('    y_true: ',y2.ravel())

print('    proba:  ',y_hat2)

print('    logloss:',loss2)



model.fit(X3,y3.ravel())

y_hat3=model.predict_proba(X3)[:,1]

loss3 =log_loss(y3,model.predict_proba(X3)[:,1])

print("model3: ")

print('    coefs:  ',model.coef_)

print('    y_true: ',y3.ravel())

print('    proba:  ',y_hat3)

print('    logloss:',loss3)



model.fit(X4,y4.ravel())

y_hat4=model.predict_proba(X4)[:,1]

loss4 =log_loss(y4,model.predict_proba(X4)[:,1])

print("model4: ")

print('    coefs:  ',model.coef_)

print('    y_true: ',y4.ravel())

print('    proba:  ',y_hat4)

print('    logloss:',loss4)

model.fit(X5,y5)

y_hat5=model.predict_proba(X5)[:,1]

loss5 =log_loss(y5,model.predict_proba(X5)[:,1])

print("model5: ")

print('    coefs:  ',model.coef_)

print('    y_true: ',y5.ravel())

print('    proba:  ',y_hat5)

print('    logloss:',loss5)



plt.title('Dataset 1,3')

plt.plot(y1,label='Y_true')

plt.plot(y_hat1,label='y_hat1 - log loss:'+str(loss1))

plt.plot(y_hat3,label='y_hat3 - log loss:'+str(loss3))

plt.legend()

plt.show()

print('X2=',X2.ravel(),'Y2=',y2.ravel())

print('X5=',X5.ravel(),'Y5=',y5.ravel())

plt.title('Dataset 2,5')

plt.plot(y2,label='Y_true')

plt.plot(y_hat2,label='y_hat2 - log loss:'+str(loss2))

plt.plot(y_hat5,label='y_hat5 - log loss:'+str(loss5))

plt.legend()

plt.show()

X1=list(range(0,1000))

y1=[0]*500 +[1]*500



X2_10000=list(range(10000,10500))

X2_0    =list(range(0,500))

X2=X2_0+X2_10000

y2=y1



print('X1=',X1)

print('y1=',y1)

print('\n\n-------------------------\n\n')

print('X2=',X2)

print('y2=',y2)
X1_v=np.vstack(X1)

model.fit(X1_v,y1)

y_hat1=model.predict(X1_v)

loss1 =log_loss(y1,y_hat1)

print("model1: ")

print('    coefs:  ',model.coef_)

print('    y_true: ',y1)

print('    proba:  ',y_hat1)

print('    logloss:',loss1)



X2_v=np.vstack(X2)

model.fit(X2_v,y2)

y_hat2=model.predict(X2_v)

loss2 =log_loss(y2,y_hat2)

print("model2: ")

print('    coefs:  ',model.coef_)

print('    y_true: ',y2)

print('    proba:  ',y_hat2)

print('    logloss:',loss2)

plt.title('Dataset with X small and X big')

plt.plot(y1,label='Y_true')

plt.plot(y_hat1,label='y_hat1 - X small - log loss:'+str(loss1),alpha=.8)

plt.plot(y_hat2,label='y_hat2 - X big - log loss:'+str(loss2),alpha=.2)

plt.legend()

plt.show()

model=DecisionTreeClassifier(criterion='gini',class_weight='balanced',max_depth=None)

x=np.vstack([3,3,3,3,3,3,3,4])

y=[0,1,0,1,0,1,0,1]



# dataset 3

print('X=',x.ravel())

print('Y=',y)

model.fit(x,y)

y_hat_both=model.predict_proba(x)

y_hat_both2=y_hat_both.copy()

y_hat_both2[:,0],y_hat_both2[:,1]=y_hat_both[:,1],y_hat_both[:,0] # inverse

y_hat_both2[7][0]=0

y_hat_both2[7][1]=1



y_hat    =y_hat_both[:,1]

y_hat_4_7=y_hat_both2[:,1]

loss =log_loss(y,y_hat)

loss1=log_loss(y,y_hat_4_7)

print("X-Y: ")

print('    depth:  ',model.tree_.max_depth)

print('    y_true: ',y)

print('    proba:  ',y_hat)

print('    logloss:',loss)



print("\nWhat happen if use 4/7 instead of 3/7? let's see log loss only, but each metric give something different")

print('    proba:  ',y_hat_4_7)

print('    logloss:',loss1)



#plot tree :)

Source( tree.export_graphviz(model, out_file=None))

# plot output

plt.title('Unbalanced')

plt.plot(y,label='Y_true')

plt.plot(y_hat,label='y_hat - log loss:'+str(loss))

plt.plot(y_hat_4_7,label='y_hat using 4/7 - log loss:'+str(loss1))

plt.legend()

plt.show()

from sklearn.metrics import roc_auc_score

print('roc=',roc_auc_score(y,y_hat)    ,', gini=',roc_auc_score(y,y_hat)*2-1)

print('roc=',roc_auc_score(y,y_hat_4_7),', gini=',roc_auc_score(y,y_hat_4_7)*2-1)



import scikitplot as skplt



skplt.metrics.plot_roc_curve(y, y_hat_both)

plt.show()

skplt.metrics.plot_roc_curve(y, y_hat_both2)

plt.show()