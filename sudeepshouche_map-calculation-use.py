import numpy as np



def MeanAveragePrecision(predictions, retrieval_solution, max_predictions=100):

    """Computes mean average precision for retrieval prediction.

    Args:

        predictions: Dict mapping test image ID to a list of strings corresponding to index image IDs.

        retrieval_solution: Dict mapping test image ID to list of ground-truth image IDs.

        max_predictions: Maximum number of predictions per query to take into account. For the Google Landmark Retrieval challenge, this should be set to 100.

    Returns:

        mean_ap: Mean average precision score (float).

    Raises:

        ValueError: If a test image in `predictions` is not included in `retrieval_solutions`.

    """

    # Compute number of test images.

    num_test_images = len(retrieval_solution.keys())



    # Loop over predictions for each query and compute mAP.

    mean_ap = 0.0

    for key, prediction in predictions.items():

        if key not in retrieval_solution:

            raise ValueError('Test image %s is not part of retrieval_solution' % key)



        # Loop over predicted images, keeping track of those which were already

        # used (duplicates are skipped).

        ap = 0.0

        already_predicted = set()

        num_expected_retrieved = min(len(retrieval_solution[key]), max_predictions)

        num_correct = 0

        for i in range(min(len(prediction), max_predictions)):

            if prediction[i] not in already_predicted:

                if prediction[i] in retrieval_solution[key]:

                    num_correct += 1

                    ap += num_correct / (i + 1)

                already_predicted.add(prediction[i])



            ap /= num_expected_retrieved

            mean_ap += ap



        mean_ap /= num_test_images

    return mean_ap

import pandas as pd



PATH = '../input/landmark-retrieval-2020/'



# Get labels (landmark ids) from train file as dict

labels = pd.read_csv(PATH+'train.csv', index_col='id')['landmark_id'].to_dict()



# Check for this image id

image_id = '0000059611c7d079'



# Dict mapping test image ID to a list of predicted strings corresponding to index image IDs.

predictions = {image_id:['111e3a18bf0e529d', '3f5f7f38ea4dca61', '6cebd3221270bcc3', 'fb09f1e98c6d2f70']}



# Dict mapping test image ID to list of ground-truth image IDs.

retrieval_solution = {image_id: [k for k,v in labels.items() if v == labels[image_id]]}



map_score = MeanAveragePrecision(predictions, retrieval_solution)

map_score
