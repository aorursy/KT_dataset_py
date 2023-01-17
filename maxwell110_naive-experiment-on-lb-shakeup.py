import numpy as np

import pandas as pd

import warnings

import matplotlib

import matplotlib.pyplot as plt



from iterstrat.ml_stratifiers import MultilabelStratifiedKFold



%matplotlib inline

matplotlib.style.use('ggplot')

warnings.filterwarnings("ignore", category=FutureWarning) 
train = pd.read_csv('../input/freesound-audio-tagging-2019/train_curated.csv')

test = pd.read_csv('../input/freesound-audio-tagging-2019/sample_submission.csv')

sub1 = pd.read_csv('../input/fs2019/oof_sub1.csv')

sub2 = pd.read_csv('../input/fs2019/oof_sub2.csv')
for c in test.columns[1:]:

    cc = c.replace('(', '\(').replace(')', '\)')

    train.loc[:, c] = train['labels'].str.contains(cc).astype(int)

    if (train.loc[:, c] > 1).sum():

        raise Exception(

            'label key "{}" are duplicated in train_cur !'.format(c))
train = train.query('fname != "1d44b0bd.wav"')  # remove silent audio

train.head()
sub1.head()
sub2.head()
def _one_sample_positive_class_precisions(scores, truth):

    """ Calculate precisions for each true class for a single sample.

    This metric is MAP@K like.



    Args:

      scores:

        np.array of (num_classes,) giving the individual classifier scores.

      truth:

        np.array of (num_classes,) bools indicating which classes are true.



    Returns:

      pos_class_indices:

        np.array of indices of the true classes for this sample.

      pos_class_precisions:

        np.array of precisions corresponding to each of those classes.

    """

    num_classes = scores.shape[0]

    pos_class_indices = np.flatnonzero(truth > 0)



    # Only calculate precisions if there are some true classes.

    if not len(pos_class_indices):

        return pos_class_indices, np.zeros(0)



    # Retrieval list of classes for this sample.

    retrieved_classes = np.argsort(scores)[::-1]



    # class_rankings[top_scoring_class_index] == 0 etc.

    class_rankings = np.zeros(num_classes, dtype=np.int)

    class_rankings[retrieved_classes] = range(num_classes)



    # Which of these is a true label?

    retrieved_class_true = np.zeros(num_classes, dtype=np.bool)

    retrieved_class_true[class_rankings[pos_class_indices]] = True



    # Num hits for every truncated retrieval list.

    retrieved_cumulative_hits = np.cumsum(retrieved_class_true)



    # Precision of retrieval list truncated at each hit, in order of pos_labels.

    precision_at_hits = (

            retrieved_cumulative_hits[class_rankings[pos_class_indices]] /

            (1 + class_rankings[pos_class_indices].astype(np.float))

    )

    return pos_class_indices, precision_at_hits



def calculate_per_class_lwlrap(truth, scores):

    """

    Calculate label-weighted label-ranking average precision.



    Arguments:

      truth:

        np.array of (num_samples, num_classes) giving boolean ground-truth

        of presence of that class in that sample.

      scores:

        np.array of (num_samples, num_classes) giving the classifier-under-

        test's real-valued score for each class for each sample.



    Returns:

      per_class_lwlrap:

        np.array of (num_classes,) giving the lwlrap for each class.

      weight_per_class:

        np.array of (num_classes,) giving the prior of each

        class within the truth labels.

        Then the overall unbalanced lwlrap is

        simply np.sum(per_class_lwlrap * weight_per_class).

    """

    assert truth.shape == scores.shape

    num_samples, num_classes = scores.shape



    # Space to store a distinct precision value for each class on each sample.

    # Only the classes that are true for each sample will be filled in.

    precisions_for_samples_by_classes = np.zeros((num_samples, num_classes))

    for sample_num in range(num_samples):

        pos_class_indices, precision_at_hits = (

            _one_sample_positive_class_precisions(scores[sample_num, :],

                                                  truth[sample_num, :])

        )

        precisions_for_samples_by_classes[sample_num, pos_class_indices] = (

            precision_at_hits)



    # Compute weight per class

    labels_per_class = np.sum(truth > 0, axis=0)

    weight_per_class = labels_per_class / float(np.sum(labels_per_class))



    # Form average of each column,

    # i.e. all the precisions assigned to labels in a particular class.

    per_class_lwlrap = (np.sum(precisions_for_samples_by_classes, axis=0) /

                        np.maximum(1, labels_per_class))



    return per_class_lwlrap, weight_per_class



def lwlrap(actual, pred):

    per_class_lwlrap, weight_per_class = calculate_per_class_lwlrap(actual, pred)

    return np.sum(per_class_lwlrap * weight_per_class)
train.loc[:, 'Accelerating_and_revving_and_vroom':].sum().plot(

    kind='barh', 

    title="Number of Audio Samples per Category", 

    color='deeppink', 

    figsize=(15,25));
lwlrap_1 = lwlrap(

    train.loc[:, 'Accelerating_and_revving_and_vroom':].values,

    sub1.loc[:, 'Accelerating_and_revving_and_vroom':].values

)

print('lwlrap on sub1 : {:.5f}'.format(lwlrap_1))
lwlrap_2 = lwlrap(

    train.loc[:, 'Accelerating_and_revving_and_vroom':].values,

    sub2.loc[:, 'Accelerating_and_revving_and_vroom':].values

)

print('lwlrap on sub2 : {:.5f}'.format(lwlrap_2))
sub1_diff = []

sub2_diff = []

for i in range(1000):

    seed = 2019 + i

    fname_sel = train.fname.sample(int(len(train)/4), random_state=seed)

    public_lwlrap1 = lwlrap(

        train.loc[train.fname.isin(fname_sel), 'Accelerating_and_revving_and_vroom':].values,

        sub1.loc[sub1.fname.isin(fname_sel), 'Accelerating_and_revving_and_vroom':].values

    )

    private_lwlrap1 = lwlrap(

        train.loc[~train.fname.isin(fname_sel), 'Accelerating_and_revving_and_vroom':].values,

        sub1.loc[~sub1.fname.isin(fname_sel), 'Accelerating_and_revving_and_vroom':].values

    )

    sub1_diff.append(private_lwlrap1 - public_lwlrap1)

    public_lwlrap2 = lwlrap(

        train.loc[train.fname.isin(fname_sel), 'Accelerating_and_revving_and_vroom':].values,

        sub2.loc[sub2.fname.isin(fname_sel), 'Accelerating_and_revving_and_vroom':].values

    )

    private_lwlrap2 = lwlrap(

        train.loc[~train.fname.isin(fname_sel), 'Accelerating_and_revving_and_vroom':].values,

        sub2.loc[~sub2.fname.isin(fname_sel), 'Accelerating_and_revving_and_vroom':].values

    )

    sub2_diff.append(private_lwlrap2 - public_lwlrap2)
fig, ax = plt.subplots(2, 1, figsize=(16,8))

ax[0].hist(sub1_diff, bins=25, rwidth=0.5, color='deeppink')

ax[1].hist(sub2_diff, bins=25, rwidth=0.5, color='darkslateblue')

ax[0].set_xlim(-0.04, 0.04)

ax[1].set_xlim(-0.04, 0.04)

ax[0].set_title('sub1 (higher lwlrap)')

ax[1].set_title('sub2 (lower lwlrap)')

plt.suptitle('lwlrap difference between public and private', ha='center');
pd.Series(sub1_diff).describe()
pd.Series(sub2_diff).describe()
sub1_diff = []

sub2_diff = []

for i in range(int(1000 / 4)):

    seed = 2019 + i

    mskf = MultilabelStratifiedKFold(n_splits=4, random_state=seed)

    for private_idx, public_idx in mskf.split(

        np.zeros(len(train)), train.loc[:, 'Accelerating_and_revving_and_vroom':].values):

        public_lwlrap1 = lwlrap(

            train.iloc[public_idx, 2:].values, sub1.iloc[public_idx, 1:].values

        )

        private_lwlrap1 = lwlrap(

            train.iloc[private_idx, 2:].values, sub1.iloc[private_idx, 1:].values

        )

        sub1_diff.append(private_lwlrap1 - public_lwlrap1)

        public_lwlrap2 = lwlrap(

            train.iloc[public_idx, 2:].values, sub2.iloc[public_idx, 1:].values

        )

        private_lwlrap2 = lwlrap(

            train.iloc[private_idx, 2:].values, sub2.iloc[private_idx, 1:].values

        )

        sub2_diff.append(private_lwlrap2 - public_lwlrap2)
fig, ax = plt.subplots(2, 1, figsize=(16,8))

ax[0].hist(sub1_diff, bins=25, rwidth=0.5, color='deeppink')

ax[1].hist(sub2_diff, bins=25, rwidth=0.5, color='darkslateblue')

ax[0].set_xlim(-0.04, 0.04)

ax[1].set_xlim(-0.04, 0.04)

ax[0].set_title('sub1 (higher lwlrap)')

ax[1].set_title('sub2 (lower lwlrap)')

plt.suptitle('lwlrap difference between public and private', ha='center');
pd.Series(sub1_diff).describe()
pd.Series(sub2_diff).describe()