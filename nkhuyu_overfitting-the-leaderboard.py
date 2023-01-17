import numpy as np

from sklearn.metrics import log_loss
N = 198

y_true = np.random.randint(2, size=N)
def score_submission(pred, true):

    pred = [np.max(np.min(x, 1-10**(-15)), 10**(-15)) for x in pred]

    return log_loss(true, pred, labels=[0,1])



def score_val(val, true):

    val = np.max(np.min(val, 1-10**(-15)), 10**(-15))

    return log_loss([true], [val], labels=[0,1])
print("Correct:", score_val(1,1)/N)

print("Wrong:", score_val(1,0)/N)
n_predicts = 15

def make_predict_at(n, n_predicts=10, N=N):

    pred = np.ones(N)*0.5 

    for i in range(n, n+n_predicts):

        pred[i] = np.exp(-0.5**(i+2))

    return pred
pred = make_predict_at(0, n_predicts=n_predicts)

score = score_submission(pred, y_true)

score = score*198 - score_val(0.5, 0)*(198-n_predicts)
sums = [[0]]

track_sum = [[]]

for i in range(n_predicts):

    old_sums = sums

    old_track = track_sum

    sums = []

    track_sum = []

    for s,t in zip(old_sums, old_track):

        sums.append(s-np.log(pred[i]))

        sums.append(s-np.log(1-pred[i]))

        

        track_sum.append(t+[1])

        track_sum.append(t+[0])
diff_sums = np.abs(sums-score)

sorted_sum = sorted(zip(diff_sums, track_sum))



print(sorted_sum[0][1])

print(list(y_true[:n_predicts]))
for test_n in range(10):

    y_true = np.random.randint(2, size=N)

    

    pred = make_predict_at(0, n_predicts=n_predicts)

    score = score_submission(pred, y_true)

    score = score*198 - score_val(0.5, 0)*(198-n_predicts)



    sums = [[0]]

    track_sum = [[]]

    for i in range(n_predicts):

        old_sums = sums

        old_track = track_sum

        sums = []

        track_sum = []

        for s,t in zip(old_sums, old_track):

            sums.append(s-np.log(pred[i]))

            sums.append(s-np.log(1-pred[i]))



            track_sum.append(t+[1])

            track_sum.append(t+[0])

     

    diff_sums = np.abs(sums-score)

    sorted_sum = sorted(zip(diff_sums, track_sum))

 

    print(test_n)

    print(sorted_sum[0][1])

    print(list(y_true[:n_predicts]))
y_true = 198*[1]

y_pred = 198*[0]

logloss(y_true, y_pred)
y_true = 198*[0]

y_pred = 198*[0]

logloss(y_true, y_pred)
import scipy as sp

def logloss(act, pred):

    epsilon = 1e-15

    pred = sp.maximum(epsilon, pred)

    pred = sp.minimum(1-epsilon, pred)

    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))

    ll = ll * -1.0/len(act)

    return ll
np.log2(3453877)