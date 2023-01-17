import os, numpy as np, pandas as pd, scipy.io as sio

import matplotlib.pyplot as plt

from tqdm.notebook import tqdm

from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report

from hmmlearn.hmm import GaussianHMM

from sklearn.linear_model import LinearRegression

from scipy.stats import norm



def predict(Xp, catp, thres):

    pred = np.zeros(len(Xp))

    for i in range(6): 

        bins = [-99.0] + list(np.sort(thres[i])) + [99.0]

        pred[catp==i] = np.digitize(Xp[catp==i], bins)-1

    return pred





def calculate_matrix(transition_matrix, states, number_processes):

    """

    Expand a transition matrix to model separate processes.

    If max(open_channels) = K, then we assume K 0/1 processes. 

    E.g. our data category 3 corresponds to a maximum

    of 3 open_channels, so 3 processes.

    

    We create model a combination_with_repetition(3, 4) = 20

    transition matrix. The first row & col corresponds to all

    processes being in the first hidden state (1, 1, 1). The

    second row & col corresponds to (1, 1, 2), and so on until

    (4, 4, 4).

    

    To calculate the transition probability from (1, 2, 2) to

    (1, 1, 3), we calculate P(1->1) * P(2->1) * P(2->3). But

    also for all permutations (e.g. (2, 1, 2) and (3, 1, 1)).

    In the end, we normalize our transition matrix.

    """

    # Fill in diagonals such that each row sums to 1

    for i in range(transition_matrix.shape[0]):

        transition_matrix[i, i] = 1 - np.sum(transition_matrix[i, :])



    n0 = len(states)

    new_transition_matrix = transition_matrix.copy()

    new_states = [(x,) for x in range(n0)]

    for process in range(1, number_processes):

        # We expand our current transition matrix (that models up to `process` number

        # of separate processes) its' dimensions by n0. We basically add another

        # possible state transition for a new process.

        nc = new_transition_matrix.shape[0]

        temp_transition_matrix = np.zeros((n0*nc, n0*nc))

        temp_states = []

        for i in range(n0):

            temp_states.extend([s + (i,) for s in new_states])

            for j in range(n0):

                # We add i -> j as our final transition

                temp_transition_matrix[i*nc:(i+1)*nc, j*nc:(j+1)*nc] = transition_matrix[i][j] * new_transition_matrix

              

        # We now group similar processes together to reduce our matrix. 

        # E.g. (1, 2, 3) is the same as (2, 3, 1)

        new_states = sorted(list(set([tuple(sorted(x)) for x in temp_states])))

        new_transition_matrix = np.zeros((len(new_states), len(new_states)))

        for i in range(len(new_states)):

            ix_i = [k for k, x in enumerate(temp_states) if tuple(sorted(x)) == new_states[i]]

            for j in range(len(new_states)):

                ix_j = [k for k, x in enumerate(temp_states) if tuple(sorted(x)) == new_states[j]]

                new_transition_matrix[i, j] = np.sum(temp_transition_matrix[ix_i, :][:, ix_j])

                new_transition_matrix[i, j] /= len(ix_i)

    

    new_channels = []

    for s in new_states:

        new_channels.append(sum([states[x] for x in s]))

    new_channels= np.array(new_channels)

        

    return new_transition_matrix, new_channels







def create_hmm(signal, predictions, transmat, states):

    # Linear Regression to esimate the mean signal value

    # per unique number of open channels

    means = np.ones((len(states))) * np.NaN

    for c in range(np.min(predictions), np.max(predictions) + 1):

        mu = np.nanmedian(signal[predictions == c])

        ix = np.where(states == c)[0]

        for i in ix:

            means[i] = mu

    mask = ~np.isnan(means)

    lr = LinearRegression()

    lr.fit(states[mask].reshape(-1, 1), means[mask])

    means = lr.predict(states.reshape(-1, 1))



    # Defining our HMM

    hmm = GaussianHMM(

        n_components=len(states),           # Number of hidden states

        n_iter=50,                         # Total number of iterations

        verbose=False,                       # Show logs

        algorithm='map',                    # Use maximum a posteriori instead of Viterbi

        params='stmc',                      # Optimize start probs, transmat, means, covs

        random_state=42,

        init_params='s',                    # Manually initialize all but start probabilities

        covariance_type='tied',             # Separate covariance per hidden state

        tol=0.01                            # Convergence criterion

    )



    # Initialize the parameters of our HMM

    hmm.n_features = 1

    hmm.means_ = means.reshape(-1, 1)

    covs = np.array([np.cov(signal[~np.isnan(signal)][predictions[~np.isnan(signal)] == c]) for c in states if np.sum(~np.isnan(signal) == c) > 1000])

    hmm.covars_ = np.nanmean(covs).reshape(-1, 1)

    hmm.transmat_ = transmat

    

    return hmm
def get_Ptran_cat(cat):

    if cat==0:

        mat = [[0     , 0.1713   , 0   , 0      ],

              [0.3297, 0        , 0   , 0.01381],

              [0     , 1        , 0   , 0      ],

              [0     , 0.0002686, 0   , 0      ]]

        

    elif cat==1:

        mat =  [[0     , 0.0121, 0     , 0     ],

                [0.0424, 0     , 0.2766, 0.0101],

                [0     , 0.2588, 0     , 0     ],

                [0     , 0.0239, 0     , 0     ]]

        

    elif cat<=4:

        mat =  [[0     , 0.0067, 0     , 0     ],

                [0.0373, 0     , 0.2762, 0.0230],

                [0     , 0.1991, 0     , 0     ],

                [0     , 0.0050, 0     , 0     ]]

        

    

    elif cat==5:

        EPS = 0

        mat =  [[0.        , EPS       , 0.34493706, 0.00287762, 0.00006045, EPS       ],

                [EPS       , 0.        , 0.00040108, EPS       , EPS       , EPS       ],

                [0.16435428, 0.00438756, 0.        , 0.01714043, 0.00023227, EPS       ],

                [0.02920171, 0.00080145, 0.27065939, 0.        , 0.01805161, 0.00108684],

                [0.00268151, 0.00000064, 0.06197474, 0.30666751, 0.        , 0.06625158],

                [EPS       , EPS       , 0.00000136, 0.13616454, 0.51059444, EPS       ]]



    return np.array(mat)







def get_hidden_states(cat):

    if cat!=5: return [1, 1, 0, 0]

    else: return [0, 0, 1, 2, 3, 4]



def get_Psig(signal, States, kexp):

    Psig = np.zeros((len(signal), len(States)))

    for i in range(len(Psig)):

        Psig[i] = np.exp((-(signal[i] - States)**2)/(kexp))

    return Psig





def forward(Psig, Ptran, etat_in=None, coef=1, normalize=True):

    if etat_in is None: etat_in = np.ones(Psig.shape)/Psig.shape[1]

    alpha = np.zeros(Psig.shape) # len(sig) x n_state

    etat = np.zeros(Psig.shape) # len(sig) x n_state

    C = np.zeros(Psig.shape[0]) # scale vector for each timestep

    

    etat[0] = etat_in[0]

    alpha[0] = etat_in[0]

    if normalize: 

        alpha[0] = etat_in[0]*Psig[0]

        alpha[0]/=alpha[0].sum()



    for j in range(1, Psig.shape[0]):

        etat[j] = alpha[j-1]@Ptran

        if normalize: etat[j] /= etat[j].sum()

        etat[j] = (etat[j]**coef) * ((etat_in[j])**(1-coef))

        if normalize: etat[j] /= etat[j].sum()

        alpha[j] = etat[j]  * Psig[j]

        alpha[j] /= alpha[j].sum()

    return alpha, etat





def optimize_thres_unsupervised(pred):

    y = pd.read_csv('/kaggle/input/liverpool-ion-switching/train.csv').open_channels.values

    sig = np.load(PATH+'M301_sig.npy')

    catbatches = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 2, 3, 5, 1, 4, 3, 4, 5, 2, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]

    good_id = list(range(3600000))+list(range(3900000,5000000))

    sY_all = [[0.26]*12,

              [0.26]*12,

              [0.26]*12,

              [0.22]*12, #0.22

              [0.26]*12,

              [0.26]*12,

             ]

    L = 100000

    Y = pred.copy()

    Thres = {}

    Yopt = pred.copy()



    for b in range(70):

        Thres[b] = np.zeros(12)

        Thres[b][0] = -99

        Thres[b][-1] = 99

        poscat = range(L*b, L*(b+1))

        catbatch = int(catbatches[b])

        sY = sY_all[catbatch]



        Yloc = Y[poscat]

        floc = sig[poscat]



        adaptive_sY = np.array([sY[int(np.round(item))] for item in floc])

        floc2 = floc[np.abs(floc-np.round(floc)) - adaptive_sY < 0]



        for i in range(10):

            ni = len(floc2[np.round(floc2)<=i])

            ni2 = np.round(ni*len(floc)/ max(1, len(floc2))).astype(int)

            Ys = np.concatenate([np.sort(floc), [19]])

            Thres[b][i+1] = 0.5*(Ys[max(0,ni2)]+Ys[min(len(Ys)-1,ni2)])



        for i in range(11):

            Yloc[(Yloc>=Thres[b][i])&(Yloc<Thres[b][i+1])] = i

    #         print(Yloc.max(), Yloc.min(), i)

        Yopt[poscat] = Yloc



    print(f1_score(y[good_id], Yopt[:5000000][good_id], average='macro'))

    return Yopt, Thres
PATH = '/kaggle/input/ion-cleaned-data/'

Kexp = [.103, .120, .1307, .138, .267, .105] 

Kexpp =  [1.8,  1.8,  1.8,   1.83, 1.807, 1.8]

N_PROCESSES = [1, 1, 3, 5, 10, 1]

COEFS_BACK = [1, .9192, .9192, .8792, .9022, .9192]

COEFS_FOR = [1, .8869, .8869, .8869, .8849, .8869]

COEFS_FIN = [.618, 0.50, 0.50,  0.49, 0.509, 0.50]

COEFS_FIN3 = [0.3,  0.3,  0.3, 0.35, 0.335, 0.3]

BATCHES = np.array([0, 5, 6, 10, 15, 20, 25, 30, 35, 40, 45, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 65])

CATEGORIES = np.array([1, 1, 1, 2, 3, 5, 4, 2, 3, 4, 5, 6, 3, 4, 6, 2, 5, 4, 5, 6, 3, 6, 6])



full_pred = np.zeros(7000000)

cleaned_signal = np.load(PATH+'sig_301_sinecleaned.npy')

oofs = pd.read_csv('../input/ion-oofs/YM257method2oofpredLB9446Proba/YM257method2oofpredLB9446Proba.txt', 

                   index_col=None, skiprows=1, header=None, sep=' ').values

predictions = np.argmax(oofs, axis=1)



for c in [0,1,2,3,4,5]:



    print("\nTraining cat", c)

    kexp = Kexp[c]

    kexpp = Kexpp[c]

    coefback = COEFS_BACK[c]

    coeffor = COEFS_FOR[c]

    coef_fin = COEFS_FIN[c]

    coef_fin3 = COEFS_FIN3[c]

    Ptran, States = calculate_matrix(get_Ptran_cat(c), get_hidden_states(c), N_PROCESSES[c])



    for jb, b in enumerate(BATCHES):

        if CATEGORIES[jb]!=c+1: continue

        end_b = BATCHES[jb+1] if b!=65 else 70

        sig = cleaned_signal[100000*b:100000*end_b]

        nstates = Ptran.shape[0]

        Psig = get_Psig(sig, States, kexp)

        

        if c!=5:



            alpha0, etat0 = forward(Psig, Ptran, normalize=False)

            alpha1, etat1 = forward(Psig[::-1], np.transpose(Ptran), etat_in=etat0[::-1], coef=coefback)

            alpha2, etat2 = forward(Psig, Ptran, etat_in=etat1[::-1], coef=coeffor)



            alpha3 = etat1[::-1]*etat2*Psig**kexpp

            for j, alp in enumerate(alpha3): alpha3[j] /= alp.sum()



            pred = coef_fin*(alpha1[::-1]) + (1-coef_fin-coef_fin3)*alpha2 + coef_fin3*alpha3



            full_pred[b*100000:b*100000+len(sig)] = pred@States

            print('Max/min', (pred@States).max(), (pred@States).min())

        

        else: 

            for k in range(len(sig) // 100000):

                sub_signal = sig[k*100000:(k+1)*100000]

                oof = predictions[(b+k)*100000:(b+k+1)*100000]



                hmm = create_hmm(sub_signal, oof, Ptran, States)

                hmm.fit(sub_signal.reshape(-1, 1))

                pred = hmm.predict_proba(sub_signal.reshape(-1, 1))

                full_pred[(b+k)*100000:(b+k+1)*100000] = pred @ States

                print('Max/min', (pred@States).max(), (pred@States).min())

        

Yopt, Thres = optimize_thres_unsupervised(full_pred)
Yopt_test = Yopt[5000000:]

# cattest = np.load(PATH+"cattest.npy")

# bestsub = pd.read_csv(PATH+"subM312_0.94415_cat6_updated.txt").open_channels.values # subM318_0.95481

# Yopt_test[cattest==5] = bestsub[cattest==5]

sub = pd.read_csv('/kaggle/input/liverpool-ion-switching/sample_submission.csv')

sub['open_channels'] = Yopt_test.astype(np.int8)

sub.to_csv('M318_Kha_withNewCat6.csv', index=None, float_format='%0.4f')

plt.hist(Yopt[5000000:])