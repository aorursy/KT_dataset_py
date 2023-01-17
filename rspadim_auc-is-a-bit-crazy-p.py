import numpy as np

from sklearn import metrics

import matplotlib.pyplot as plt



def e2_dp(y_true,y_scores):

    return (y_true==1)-y_scores

def e2_dn(y_true,y_scores):

    return (y_true==0)-(1-y_scores)

def e2(y_true,y_scores):

    erms=0

    EE=0

    for i in y_true:

        erms=np.sqrt((e2_dp(y_true[i],y_scores[i])**2+e2_dn(y_true[i],y_scores[i])**2)/2)

        EE+=erms

    return EE/len(y_true)

    

def p(y_true,y_scores):

    plt.title("AUC*2-1="+str(metrics.roc_auc_score(y_true, y_scores)*2-1)+"\n"+

              "LOGLOSS="+str(metrics.log_loss(y_true,y_scores))+"\n"+

              "HINGE="+str(metrics.hinge_loss((y_true-.5)*2,(y_scores-.5)*2))+"\n"+

              "E2="+str(e2(y_true,y_scores))

             )

    plt.plot(y_true,label='y_trye')

    plt.plot(y_scores,label='y_scores')

    plt.legend()

    plt.show()

    
y_true = np.array([0,0,0,0,0,0,0,0,0,0,1])

y_scores = np.array([0,0,0,0,0,0,0,0,0,0,1])

p(y_true,y_scores)
y_true = np.array([0,0,0,0,0,0,0,0,0,0,1])

y_scores = np.array([0,0,0,0,0,0,0,0,0,0,0.49])

p(y_true,y_scores)
y_true = np.array([0,0,0,0,0,0,0,0,0,0,1])

y_scores = np.array([.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5])

p(y_true,y_scores)
y_true = np.array([1,1,0,0,0,0,0,0,0,1,1])

y_scores = np.array([.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5])

p(y_true,y_scores)
y_true = np.array([1,1,0,0,0,0,0,0,0,1,1])

y_scores = np.array([.75,.75,.75,.75,.75,.75,.75,.75,.75,.75,.75])

p(y_true,y_scores)
y_true = np.array([1,1,0,0,0,0,0,0,0,0,0])

y_scores = np.array([.75,.75,.75,.75,.75,.75,.75,.75,.75,.75,.75])

p(y_true,y_scores)
y_true = np.array([1,0,0,0,1,0,1,0,0,1,1])

y_scores = np.array([.75,.75,.75,.75,.75,.75,.75,.75,.75,.75,.75])

p(y_true,y_scores)
y_true = np.array([1,0,0,0,0,0,0,0,0,0,0])

y_scores = np.array([1,.9,.8,.7,.6,.5,.4,.3,.2,.1,0])

p(y_true,y_scores)
y_true = np.array([0,0,0,0,0,0,0,0,0,0,1])

y_scores = np.array([1,.9,.8,.7,.6,.5,.4,.3,.2,.1,0])

p(y_true,y_scores)
y_true = np.array([1,1,1,1,1,1,1,1,1,1,0])

y_scores = np.array([1,.9,.8,.7,.6,.5,.4,.3,.2,.1,0])

p(y_true,y_scores)