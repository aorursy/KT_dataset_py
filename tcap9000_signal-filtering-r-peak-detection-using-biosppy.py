import numpy as np 
import pandas as pd 
import os
from biosppy import storage
from biosppy.signals import ecg
import matplotlib.pyplot as plt
import numpy as np
import pdb
import pywt

def upArrow_op(li, j):
    if j == 0:
        return [1]
    N = len(li)
    li_n = np.zeros(2 ** (j - 1) * (N - 1) + 1)
    for i in range(N):
        li_n[2 ** (j - 1) * i] = li[i]
    return li_n


def period_list(li, N):
    n = len(li)
    # append [0 0 ...]
    n_app = N - np.mod(n, N)
    li = list(li)
    li = li + [0] * n_app
    if len(li) < 2 * N:
        return np.array(li)
    else:
        li = np.array(li)
        li = np.reshape(li, [-1, N])
        li = np.sum(li, axis=0)
        return li


def circular_convolve_mra(h_j_o, w_j):
    ''' calculate the mra D_j'''
    N = len(w_j)
    l = np.arange(N)
    D_j = np.zeros(N)
    for t in range(N):
        index = np.mod(t + l, N)
        w_j_p = np.array([w_j[ind] for ind in index])
        D_j[t] = (np.array(h_j_o) * w_j_p).sum()
    return D_j


def circular_convolve_d(h_t, v_j_1, j):
    '''
    jth level decomposition
    h_t: \tilde{h} = h / sqrt(2)
    v_j_1: v_{j-1}, the (j-1)th scale coefficients
    return: w_j (or v_j)
    '''
    N = len(v_j_1)
    L = len(h_t)
    w_j = np.zeros(N)
    l = np.arange(L)
    for t in range(N):
        index = np.mod(t - 2 ** (j - 1) * l, N)
        v_p = np.array([v_j_1[ind] for ind in index])
        w_j[t] = (np.array(h_t) * v_p).sum()
    return w_j


def circular_convolve_s(h_t, g_t, w_j, v_j, j):
    '''
    (j-1)th level synthesis from w_j, w_j
    see function circular_convolve_d
    '''
    N = len(v_j)
    L = len(h_t)
    v_j_1 = np.zeros(N)
    l = np.arange(L)
    for t in range(N):
        index = np.mod(t + 2 ** (j - 1) * l, N)
        w_p = np.array([w_j[ind] for ind in index])
        v_p = np.array([v_j[ind] for ind in index])
        v_j_1[t] = (np.array(h_t) * w_p).sum()
        v_j_1[t] = v_j_1[t] + (np.array(g_t) * v_p).sum()
    return v_j_1


def modwt(x, filters, level):
    '''
    filters: 'db1', 'db2', 'haar', ...
    return: see matlab
    '''
    # filter
    wavelet = pywt.Wavelet(filters)
    h = wavelet.dec_hi
    g = wavelet.dec_lo
    h_t = np.array(h) / np.sqrt(2)
    g_t = np.array(g) / np.sqrt(2)
    wavecoeff = []
    v_j_1 = x
    for j in range(level):
        w = circular_convolve_d(h_t, v_j_1, j + 1)
        v_j_1 = circular_convolve_d(g_t, v_j_1, j + 1)
        wavecoeff.append(w)
    wavecoeff.append(v_j_1)
    return np.vstack(wavecoeff)


def imodwt(w, filters):
    ''' inverse modwt '''
    # filter
    wavelet = pywt.Wavelet(filters)
    h = wavelet.dec_hi
    g = wavelet.dec_lo
    h_t = np.array(h) / np.sqrt(2)
    g_t = np.array(g) / np.sqrt(2)
    level = len(w) - 1
    v_j = w[-1]
    for jp in range(level):
        j = level - jp - 1
        v_j = circular_convolve_s(h_t, g_t, w[j], v_j, j + 1)
    return v_j


def modwtmra(w, filters):
    ''' Multiresolution analysis based on MODWT'''
    # filter
    wavelet = pywt.Wavelet(filters)
    h = wavelet.dec_hi
    g = wavelet.dec_lo
    # D
    level, N = w.shape
    level = level - 1
    D = []
    g_j_part = [1]
    for j in range(level):
        # g_j_part
        g_j_up = upArrow_op(g, j)
        g_j_part = np.convolve(g_j_part, g_j_up)
        # h_j_o
        h_j_up = upArrow_op(h, j + 1)
        h_j = np.convolve(g_j_part, h_j_up)
        h_j_t = h_j / (2 ** ((j + 1) / 2.))
        if j == 0: h_j_t = h / np.sqrt(2)
        h_j_t_o = period_list(h_j_t, N)
        D.append(circular_convolve_mra(h_j_t_o, w[j]))
    # S
    j = level - 1
    g_j_up = upArrow_op(g, j + 1)
    g_j = np.convolve(g_j_part, g_j_up)
    g_j_t = g_j / (2 ** ((j + 1) / 2.))
    g_j_t_o = period_list(g_j_t, N)
    S = circular_convolve_mra(g_j_t_o, w[-1])
    D.append(S)
    return np.vstack(D)

def fix_broken_csv(file_name):
    rf = open (os.path.join('../input',file_name), 'r')
    ff = open(os.path.join('output',file_name), 'w')
    line = rf.readline()
    ff.write(line)
    while line:
        line = rf.readline()
        line = line.replace(',AMEER','')
        line = line.replace(',j',',0')
        if "03:51:35.125749,52303:52:33.684308,664" in line:
            ff.write("03:51:35.125749,523\n")
            ff.write("03:52:33.684308,664\n")   
        else:
            ff.write(line)
    rf.close()
    ff.close()
os.mkdir('output')
all_files = os.listdir("../input")
all_files = np.array(all_files)[["ppg" in x for x in  all_files]]
[fix_broken_csv(f) for f in all_files]
def load_gamer(x):
    df = pd.read_csv('output/gamer'+x+'-ppg-2000-01-01.csv')
    df['Day'] = len(df)*[1]
    df2 = pd.read_csv('output/gamer'+x+'-ppg-2000-01-02.csv')
    df2['Day']= len(df2)*[2]
    df = df.append(df2, ignore_index=True) 
    return df
def peaks_to_series(signal, peaks):
    pks=np.zeros(len(signal))
    pks[peaks]=1
    return pks
def filter_signal(signal, sampling_rate):
    order = int(0.3 * sampling_rate)
    filtered, _, _ = ecg.st.filter_signal(signal=signal,
                                      ftype='FIR',
                                      band='bandpass',
                                      order=order,
                                      frequency=[3, 45],
                                      sampling_rate=sampling_rate)
    
    rpeaks, = ecg.hamilton_segmenter(signal=filtered, sampling_rate=sampling_rate)

    
    rpeaks_corrected, = ecg.correct_rpeaks(signal=filtered,
                             rpeaks=rpeaks,
                             sampling_rate=sampling_rate,
                             tol=0.05)
    
    rpeaks = peaks_to_series(signal,rpeaks)
    rpeaks_corrected = peaks_to_series(signal,rpeaks_corrected)
    
    return filtered, rpeaks, rpeaks_corrected
def process_gamer(x):
    print("Processing Gamer #", x)
    df = load_gamer(x)
    signal = df['Red_Signal'].values
    out, rpeak, rpeak_corr = filter_signal(signal, sampling_rate=100.)
    df['ecg_out']=out
    df['rpeak']=rpeak
    df['rpeak_corr']=rpeak_corr
    df.to_csv('output/gamer-'+x+'-ecg.csv', index=False)
    del df
    del signal
    del out
[process_gamer(gamer) for  gamer in ["1","2","3","4","5"]]
df=pd.read_csv('output/gamer-1-ecg.csv') # try any of the GAMERS  
minr=3000
maxr=6000
number_of_freq=5
ecgd=df['ecg_out'].values[minr:maxr]
raw=df['Red_Signal'].values[minr:maxr]
rp=df['rpeak'].values[minr:maxr]
wt = modwt(ecgd, 'db2', number_of_freq)
wtmra = modwtmra(wt, 'db2')
for i in range(len(wtmra)):
    print ("Wavlet Transfom ", i )
    plt.plot(wtmra[i])
    plt.show()    

print ("de-noised signal" )
plt.plot(wtmra[4] + wtmra[5])
plt.show()       
    
print ("Filtered Signal from ECG module")    
plt.plot(ecgd)
plt.show()

print ("Raw Signal")    
plt.plot(raw)
plt.show()

print ("R-Peaks from ECG module")
plt.plot(raw*rp)
plt.show()
