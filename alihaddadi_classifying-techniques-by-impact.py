import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

%pylab inline



from tqdm import tqdm

from scipy.fftpack import fft

from sklearn.ensemble import RandomForestClassifier

from sklearn.decomposition import KernelPCA, PCA

from sklearn.multiclass import OneVsRestClassifier

from sklearn.model_selection import cross_val_predict

from sklearn.metrics import confusion_matrix, classification_report

from sklearn.preprocessing import LabelEncoder



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
def running_mean(x, N):

    cumsum = numpy.cumsum(numpy.insert(x, 0, 0)) 

    return (cumsum[N:] - cumsum[:-N]) / N 



def get_cm(kernel, data):

    "Get the confusion matrix and it's heatmap"

    le = LabelEncoder()

    x, y = data.drop('Target', axis=1), le.fit_transform(data.Target)

    

    estimator = OneVsRestClassifier(RandomForestClassifier(n_jobs=-1,

                                                           n_estimators=500))

    pca = KernelPCA(kernel=kernel)

    x = pca.fit_transform(x)

    

    p = cross_val_predict(estimator, x, y)

    cm = pd.DataFrame(confusion_matrix(y, p))

    cm.index, cm.columns = le.classes_, le.classes_

    

    cm = cm / cm.sum(axis=1)

    sns.heatmap(cm, cmap='gray_r', annot=True)

    plt.title('With KernelPCA({})'.format(kernel))

    print(classification_report(y, p))

    

df = pd.read_csv('../input/Taekwondo_Technique_Classification_Stats.csv', index_col=0)

df.info()
temp = []

for technique in 'RBCP':

    t = df[[col for col in df.columns if technique+'.' in col]][2:].astype(float)

    temp.append(t)

R, B, C, P = temp
window = 5

plt.subplots(figsize=(10, 25))

ax1 = plt.subplot(29, 4, 1)

plt.gca().get_yaxis().set_ticklabels([])

plt.gca().get_xaxis().set_ticklabels([])

index = 1

for col in R.columns:

    plt.subplot(29, 4, index, sharex=ax1)

    running = running_mean(R[col].values, window)

    threshold = R[col].mean()

    plt.plot(running[1100: 3000], color='black', alpha=1, linewidth=0.7)

    plt.plot([threshold for _ in running[1100:3000]], color='green', linewidth=0.8)

    plt.gca().get_yaxis().set_ticklabels([])

    

    col = col.replace('R', 'B')

    plt.subplot(29, 4, index+1, sharex=ax1)

    try:

        running = running_mean(B[col].values, window)

        threshold = B[col].mean()

    except:

        running = [R[col.replace('B', 'R')].mean() for _ in range(1000)]

        threshold = R[col.replace('B', 'R')].mean()

    plt.plot(running[1100: 3000], color='red', alpha=1, linewidth=0.7)

    plt.plot([threshold for _ in running[1100:3000]], color='green', linewidth=0.8)

    plt.gca().get_yaxis().set_ticklabels([])

    

    col = col.replace('B', 'C')

    plt.subplot(29, 4, index+2, sharex=ax1)

    running = running_mean(C[col].values, window)

    threshold = C[col].mean()

    plt.plot(running[1100: 3000], color='orange', alpha=1, linewidth=0.7)

    plt.plot([threshold for _ in running[1100:3000]], color='green', linewidth=0.8)

    plt.gca().get_yaxis().set_ticklabels([])

    

    col = col.replace('C', 'P')

    plt.subplot(29, 4, index+3, sharex=ax1)

    running = running_mean(P[col].values, window)

    threshold = P[col].mean()

    plt.plot(running[1100: 3000], color='blue', alpha=1, linewidth=0.7)

    plt.plot([threshold for _ in running[1100:3000]], color='green', linewidth=0.8)

    plt.gca().get_yaxis().set_ticklabels([])

    index += 4

colors = 'Black: Roundhouse, Red: Back Kick, Orange: Cut, Blue: Punch'

plt.suptitle('{}\n (Rolling mean of signal with {} window)'.format(colors, window), fontsize=20)

plt.subplots_adjust(top=0.95)
# Performing FFT on the data

ndf, labels = [], []

for consider, name in zip([R, B, C, P], ['Roundhouse', 'Back Kick', 'Cut', 'Punch']):

    for col in consider.columns:

        meanval = consider[col].mean()

        signal = consider[col].fillna(meanval)[1100: 3000]

        f = fft(signal)

        power = np.concatenate([f.real, f.imag])

        power = np.sqrt(np.power(f.real, 2) + np.power(f.imag, 2))

        ndf.append(power)

        labels.append(name)

print(set(list(map(len, ndf))), 'values are given by the FFT')
# Preparing the data for classification

data = pd.DataFrame(ndf)

data['Target'] = labels

print(data.shape, 'Data shape')


# Let's see what vanilla PCA does.

le = LabelEncoder()

x, y = data.drop('Target', axis=1), le.fit_transform(data.Target)



print(x.shape, 'shape before PCA')

pca = PCA()

x = pca.fit_transform(x)

print(x.shape, 'shape after PCA')
# % Variance explained

plt.plot(np.cumsum(pca.explained_variance_ratio_), '.-', label='Variance explained')

plt.plot([0, 120], [1, 1], label='1', color='black')

plt.title('Variance explained with components')

plt.legend()
# get_cm('linear', data)

# get_cm('poly', data)

# get_cm('cosine', data)

# get_cm('sigmoid', data)

get_cm('rbf', data)
df.T.head()
# We pre process the data a little.

dft = df.T.copy()

means = dft.drop(['ID', 'Trial #'], axis=1).astype(float).mean(axis=1)

temp = dft.drop(['ID', 'Trial #'], axis=1).astype(float).apply(lambda x: abs(x-1026), axis=1)



for i in range(11):

    temp = temp.replace(i, np.nan)



temp.columns = [str(i) for i, v in enumerate(temp.columns)]

cols_to_drop = []

for col in temp.columns:

    if temp[col].var() == 0:

        cols_to_drop.append(col)

print('{} columns present before drop'.format(temp.shape[1]))

temp = temp.drop(cols_to_drop, axis=1)

print('{} columns present after drop'.format(temp.shape[1]))
# We measure Mean and Max signal strengths

temp['Mean_hits'] = temp.mean(axis=1)  # THIS IS WHAT IS BEING MEASURED

temp['Max_hits'] = temp.max(axis=1)  # THIS IS WHAT IS BEING MEASURED

temp['ID'] = dft.ID

temp['Technique'] = dft.index.str[0]



plt.subplots(figsize=(10, 5))

plt.subplot(1, 2, 1)

sns.violinplot(x='ID', y='Mean_hits', data=temp, linewidth=1)

plt.subplot(1, 2, 2)

sns.violinplot(x='ID', y='Max_hits', data=temp, linewidth=1)



plt.suptitle('Signal strength grouped by player IDs')
ct = pd.crosstab(temp['ID'], temp['Technique'],

                        values=temp['Max_hits'], aggfunc=np.mean)

ct
sns.heatmap(ct / ct.sum(), cmap='gray_r', annot=True, linewidth=0.5, linecolor='black')

sns.plt.title('Who hits the hardest per technique? (Do not compare across rows)')
sns.heatmap((ct.T / ct.sum(axis=1)).T, cmap='gray_r',

            annot=True, linewidth=0.5, linecolor='black')

sns.plt.title("Which technique is a person's best? (Do not compare across cols)" )
# Let's add player info now.

players = pd.read_csv('../input/Table1.csv')

players
temp['Participant ID'] = temp['ID']

temp2 = pd.merge(temp, players, on='Participant ID', how='left')



sns.factorplot(x='Age', y='Max_hits', data=temp2)
sns.factorplot(x='Weight (kg)', y='Max_hits', data=temp2)
temp2[['Weight (kg)', 'Age', 'Max_hits']].corr()