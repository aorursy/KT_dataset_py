import pandas as pd

import numpy as np



def compare(results, baseline_tactic, current_tactic):

    """This function compares the results between a current tactic and a baseline.

    For each result in the current tactic it compares the result and the score.



    :param results: DataFrame with results.

    :param baseline_tactic: Tactic that will be used as a baseline.

    :param current_tactic: Tactic with the results to compare.

    :return: DataFrame with model, and differences in result and score between current and baseline tactics.

    """

    comparison = pd.DataFrame(columns=['Model',

                                       'Result',

                                       'Score'])



    for row in results.query('Tactic == ' + str(current_tactic)).itertuples(index=False):

        previous = results.query('Tactic == ' + str(baseline_tactic) + ' and Model == "' + row.Model + '"')

        comparison = comparison.append({

            'Model': row.Model,

            'Result': '{:.2%}'.format((row.Result - float(previous.Result)) / float(previous.Result)),

            'Score': '{:.2%}'.format((row.Score - float(previous.Score)) / float(previous.Score))

        }, ignore_index=True)



    return comparison



results = pd.read_csv('../input/tactic-98-results/results.csv', index_col='Id', engine='python')
compare(results, 1, 3)
compare(results, 3, 5)
compare(results, 3, 6)
compare(results, 6, 7)
compare(results, 6, 8)
compare(results, 6, 9)
compare(results, 6, 10)