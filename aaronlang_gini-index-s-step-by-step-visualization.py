import pandas as pd 

import numpy as np 

def gini(actual, pred, cmpcol = 0, sortcol = 1):

     assert( len(actual) == len(pred) )

     all = np.asarray(np.c_[ actual, pred, np.arange(len(actual)) ], dtype=np.float)

     all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]

     totalLosses = all[:,0].sum()

     giniSum = all[:,0].cumsum().sum() / totalLosses

     giniSum -= (len(actual) + 1) / 2.

     return giniSum / len(actual)

 

def gini_normalized(a, p):

    return gini(a, p) / gini(a, a)
# create data input 

actual = np.array([1, 0, 0, 1, 0,

                   0, 0, 0, 1, 1,

                   1, 1, 1, 1, 1,

                   1, 0, 0, 0, 0])  # actual value of target variable taking on 0/1



predict = np.array([0.95, 0.40, 0.60, 0.28, 0.90,

                    0.87, 0.30, 0.20, 0.51, 0.70,

                    0.90, 0.60, 0.21, 0.53, 0.97,

                    0.93, 0.89, 0.25, 0.34, 0.21]) # predict the probability that target value is 1
# sort actual values based on predicted probability in descending order 

actual_predict = actual[np.argsort(-predict)]



# sort actual values with descending order (i.e. an optimal solution)

actual_optimal = np.sort(actual)[::-1]
import matplotlib.pyplot as plt 

%matplotlib inline 



plt.figure()

ax = plt.subplot(111)

plt.plot(np.arange(0,21),np.append(0,actual_predict.cumsum()))

plt.scatter(np.arange(1,21), actual_predict.cumsum())

plt.plot([0, 20], [0, actual_predict.cumsum()[-1]], 'k-', lw=1)

plt.plot([0, 20],[0,0],'k-',lw=1)

plt.plot([20, 20],[0,actual_predict.cumsum()[-1]],'k-',lw=1)

plt.xticks(np.arange(0,21))

plt.yticks(np.arange(0,actual_predict.cumsum()[-1]+1))

plt.xlabel('[Figure 1] accumulated number of samples')

plt.ylabel('pct of accumulated positive values')



ax.set_yticklabels(labels=(np.arange(0,1.1,0.1)))



for spine in plt.gca().spines.values():

    spine.set_visible(False)

plt.tick_params(top='off', bottom='off', 

                left='off', right='off')



for i in range(20): 

    plt.plot([i, i+1],[actual_predict.cumsum()[i],actual_predict.cumsum()[i]],'--',c='red')

    plt.plot([i,i],[0,actual_predict.cumsum()[i]],'--',c='red')

    plt.fill_between([i, i+1],[actual_predict.cumsum()[i],actual_predict.cumsum()[i]]

                     ,[0,0],

                     facecolor='red', alpha = 0.5)
plt.figure()

ax = plt.subplot(111)

plt.plot(np.arange(0,21),np.append(0,actual_predict.cumsum()))

plt.scatter(np.arange(1,21), actual_predict.cumsum())

plt.plot([0, 20], [0, actual_predict.cumsum()[-1]], 'k-', lw=1)

plt.plot([0, 20],[0,0],'k-',lw=1)

plt.plot([20, 20],[0,actual_predict.cumsum()[-1]],'k-',lw=1)

plt.xticks(np.arange(0,21,2))

plt.yticks(np.arange(0,actual_predict.cumsum()[-1]+1))

plt.xlabel('[Figure 2] pct of accumulated samples')

plt.ylabel('pct of accumulated positive values')



ax.set_yticklabels(labels=(np.arange(0,1.1,0.1)))

ax.set_xticklabels(labels=(np.arange(0,1.1,0.1)))



for spine in plt.gca().spines.values():

    spine.set_visible(False)

plt.tick_params(top='off', bottom='off', 

                left='off', right='off')



plt.fill_between(np.arange(0,21),

                 np.append(0,actual_predict.cumsum()),

                 np.arange(0,21)*(actual_predict.cumsum()[-1]/20),

                 facecolor='red', alpha = 0.5)





for i in range(20): 

    plt.plot([i, i+1],[actual_predict.cumsum()[i],actual_predict.cumsum()[i]],'--',c='red')

    plt.plot([i,i],[0,actual_predict.cumsum()[i]],'--',c='red')



plt.figure()

ax = plt.subplot(111)

plt.plot(np.arange(0,21),np.append(0,actual_optimal.cumsum()))

plt.scatter(np.arange(1,21), actual_optimal.cumsum())

plt.plot([0, 20], [0, actual_optimal.cumsum()[-1]], 'k-', lw=1)

plt.plot([0, 20],[0,0],'k-',lw=1)

plt.plot([20, 20],[0,actual_optimal.cumsum()[-1]],'k-',lw=1)

plt.xticks(np.arange(0,21,2))

plt.yticks(np.arange(0,actual_optimal.cumsum()[-1]+1))

plt.xlabel('[Figure 3] pct of accumulated samples')

plt.ylabel('pct of accumulated positive values')



ax.set_yticklabels(labels=(np.arange(0,1.1,0.1)))

ax.set_xticklabels(labels=(np.arange(0,1.1,0.1)))



for spine in plt.gca().spines.values():

    spine.set_visible(False)

plt.tick_params(top='off', bottom='off', 

                left='off', right='off')



plt.fill_between(np.arange(0,21),

                 np.append(0,actual_optimal.cumsum()),

                 np.arange(0,21)*(actual_optimal.cumsum()[-1]/20),

                 facecolor='red', alpha = 0.5)





for i in range(20): 

    plt.plot([i, i+1],[actual_optimal.cumsum()[i],actual_optimal.cumsum()[i]],'--',c='red')

    plt.plot([i,i],[0,actual_optimal.cumsum()[i]],'--',c='red')



plt.figure()

ax = plt.subplot(111)

plt.plot(np.arange(0,21),np.append(0,actual_predict.cumsum()))

plt.scatter(np.arange(1,21), actual_predict.cumsum())

plt.plot([0, 20], [0, actual_predict.cumsum()[-1]], 'k-', lw=1)

plt.plot([0, 20],[0,0],'k-',lw=1)

plt.plot([20, 20],[0,actual_predict.cumsum()[-1]],'k-',lw=1)

plt.xticks(np.arange(0,21,2))

plt.yticks(np.arange(0,actual_predict.cumsum()[-1]+1))

plt.xlabel('[Figure 4] pct of accumulated samples')

plt.ylabel('pct of accumulated positive values')



ax.set_yticklabels(labels=(np.arange(0,1.1,0.1)))

ax.set_xticklabels(labels=(np.arange(0,1.1,0.1)))



for spine in plt.gca().spines.values():

    spine.set_visible(False)

plt.tick_params(top='off', bottom='off', 

                left='off', right='off')



plt.plot(np.arange(0,21),np.append(0,actual_optimal.cumsum()))

plt.scatter(np.arange(1,21), actual_optimal.cumsum())



plt.fill_between(np.arange(0,21),

                 np.append(0,actual_optimal.cumsum()),

                 np.append(0,actual_predict.cumsum()),

                 facecolor='blue',alpha=0.5)



plt.fill_between(np.arange(0,21),

                 np.append(0,actual_predict.cumsum()),

                 np.arange(0,21)*(actual_predict.cumsum()[-1]/20),

                 facecolor='red',alpha=0.5)

plt.show()