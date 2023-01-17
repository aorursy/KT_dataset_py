# ライブラリ読み込み

import numpy as np

import sklearn.metrics
# LwLRAP計算関数

# from official code https://colab.research.google.com/drive/1AgPdhSp7ttY18O3fEoHOQKlt_3HJDLi8#scrollTo=cRCaCIb9oguU

def _one_sample_positive_class_precisions(scores, truth):

    """Calculate precisions for each true class for a single sample.



    Args:

      scores: np.array of (num_classes,) giving the individual classifier scores.

      truth: np.array of (num_classes,) bools indicating which classes are true.



    Returns:

      pos_class_indices: np.array of indices of the true classes for this sample.

      pos_class_precisions: np.array of precisions corresponding to each of those

        classes.

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

            (1 + class_rankings[pos_class_indices].astype(np.float)))

    return pos_class_indices, precision_at_hits





def calculate_per_class_lwlrap(truth, scores):

    """Calculate label-weighted label-ranking average precision.



    Arguments:

      truth: np.array of (num_samples, num_classes) giving boolean ground-truth

        of presence of that class in that sample.

      scores: np.array of (num_samples, num_classes) giving the classifier-under-

        test's real-valued score for each class for each sample.



    Returns:

      per_class_lwlrap: np.array of (num_classes,) giving the lwlrap for each

        class.

      weight_per_class: np.array of (num_classes,) giving the prior of each

        class within the truth labels.  Then the overall unbalanced lwlrap is

        simply np.sum(per_class_lwlrap * weight_per_class)

    """

    assert truth.shape == scores.shape

    num_samples, num_classes = scores.shape

    # Space to store a distinct precision value for each class on each sample.

    # Only the classes that are true for each sample will be filled in.

    precisions_for_samples_by_classes = np.zeros((num_samples, num_classes))

    for sample_num in range(num_samples):

        pos_class_indices, precision_at_hits = (

            _one_sample_positive_class_precisions(scores[sample_num, :],

                                                  truth[sample_num, :]))

        precisions_for_samples_by_classes[sample_num, pos_class_indices] = (

            precision_at_hits)

    labels_per_class = np.sum(truth > 0, axis=0)

    weight_per_class = labels_per_class / float(np.sum(labels_per_class))

    # Form average of each column, i.e. all the precisions assigned to labels in

    # a particular class.

    per_class_lwlrap = (np.sum(precisions_for_samples_by_classes, axis=0) /

                        np.maximum(1, labels_per_class))

    # overall_lwlrap = simple average of all the actual per-class, per-sample precisions

    #                = np.sum(precisions_for_samples_by_classes) / np.sum(precisions_for_samples_by_classes > 0)

    #           also = weighted mean of per-class lwlraps, weighted by class label prior across samples

    #                = np.sum(per_class_lwlrap * weight_per_class)

    return per_class_lwlrap, weight_per_class

# 実際に計算してみる。

y_true = np.array([1, 0, 0,])

y_score = np.array([0.7, 0.1, 0.2])

pos_class_indices, precision_at_hits = _one_sample_positive_class_precisions(y_score, y_true)

print("正解ラベル", pos_class_indices)

print("スコア", precision_at_hits)
# 実際に計算してみる。

y_true = np.array([1, 0, 0,])

y_score = np.array([0.1, 0.7, 0.2])

pos_class_indices, precision_at_hits = _one_sample_positive_class_precisions(y_score, y_true)

print("正解ラベル", pos_class_indices)

print("スコア", precision_at_hits)
# 実際に計算してみる。

y_true = np.array([1, 0, 1,])

y_score = np.array([0.7, 0.1, 0.2])

pos_class_indices, precision_at_hits = _one_sample_positive_class_precisions(y_score, y_true)

print("正解ラベル", pos_class_indices)

print("スコア", precision_at_hits)
# 実際に計算してみる。

y_true = np.array([1, 0, 1,])

y_score = np.array([0.1, 0.7, 0.2])

pos_class_indices, precision_at_hits = _one_sample_positive_class_precisions(y_score, y_true)

print("正解ラベル", pos_class_indices)

print("スコア", precision_at_hits)
# 実際に計算してみる。

y_true = np.array([[1, 0, 1,], [0, 1, 1]])

y_score = np.array([[0.1, 0.7, 0.2], [0.1, 0.7, 0.2]])

_, precision_at_hits1 = _one_sample_positive_class_precisions(y_score[0], y_true[0])

print("sample 1 のスコア", precision_at_hits1)

_, precision_at_hits2 = _one_sample_positive_class_precisions(y_score[1], y_true[1])

print("sample 2 のスコア", precision_at_hits2)

score, weight = calculate_per_class_lwlrap(y_true, y_score)

print("各クラスのスコア", score)

print("各クラスの重み", weight)

LwLRAP = (score*weight).sum()

print("LwLRAP", LwLRAP)