from os import listdir

from os.path import join



import numpy as np





def compute_det_curve(target_scores, nontarget_scores):



    n_scores = target_scores.size + nontarget_scores.size

    all_scores = np.concatenate((target_scores, nontarget_scores))

    labels = np.concatenate((np.ones(target_scores.size), np.zeros(nontarget_scores.size)))



    # Sort labels based on scores

    indices = np.argsort(all_scores, kind='mergesort')

    labels = labels[indices]



    # Compute false rejection and false acceptance rates

    tar_trial_sums = np.cumsum(labels)

    nontarget_trial_sums = nontarget_scores.size - (np.arange(1, n_scores + 1) - tar_trial_sums)



    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / target_scores.size))  # false rejection rates

    far = np.concatenate((np.atleast_1d(1), nontarget_trial_sums / nontarget_scores.size))  # false acceptance rates

    thresholds = np.concatenate((np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))  # Thresholds are the sorted scores



    return frr, far, thresholds





def compute_eer(target_scores, nontarget_scores):

    """ Returns equal error rate (EER) and the corresponding threshold. """

    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)

    abs_diffs = np.abs(frr - far)

    min_index = np.argmin(abs_diffs)

    eer = np.mean((frr[min_index], far[min_index]))

    return eer, thresholds[min_index]





def log_loss(y_true, y_pred, eps=1e-15):



    y_true = y_true[:, np.newaxis]

    y_pred = y_pred[:, np.newaxis]



    y_true = np.append(1 - y_true, y_true, axis=1)

    y_pred = np.append(1 - y_pred, y_pred, axis=1)



    y_pred = np.clip(y_pred, eps, 1 - eps)



    y_pred /= y_pred.sum(axis=1)[:, np.newaxis]

    loss = -(y_true * np.log(y_pred)).sum(axis=1)



    return np.average(loss)
methods_name = ['Deepfakes', 'FaceSwap', 'Face2Face', 'NeuralTextures']

methods_size = []

submission = []



videos = '/kaggle/input/faceforensics/original_sequences/youtube/c23/videos'

files = [join(videos, filename) for filename in listdir(videos)]

methods_size.append(len(files))

submission.extend(sorted(files))

print(len(files), 'Original')



for method in methods_name:

    videos = f'/kaggle/input/faceforensics/manipulated_sequences/{method}/c23/videos'

    files = [join(videos, filename) for filename in listdir(videos)]

    methods_size.append(len(files))

    submission.extend(sorted(files))

    print(len(files), method)
labels = np.ones(len(submission), dtype=np.float32) / 2
offset = methods_size[0]

original_labels = labels[:offset]

for method, size in zip(methods_name, methods_size[1:]):

    fake_labels = labels[offset:offset + size]

    eer, _ = compute_eer(fake_labels, original_labels)

    ll = log_loss(np.concatenate([np.zeros_like(original_labels), np.ones_like(fake_labels)]),

                  np.concatenate([original_labels, fake_labels]))

    print(f'{method:15s} EER {eer:.4f} LL {ll:.4f}')

    offset += size