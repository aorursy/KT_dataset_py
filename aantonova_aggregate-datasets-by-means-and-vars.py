import numpy as np
import pandas as pd
import time
import warnings
warnings.simplefilter(action = 'ignore')
import csv
general_columns = ['object_id', 'count']
mean_columns = ['mjd', 'flux', 'flux_err', 'detected']
var_columns = ['mjd', 'flux', 'flux_err']
all_columns = general_columns + \
              [c + '_all_mean' for c in mean_columns] + \
              [c + '_pb' + str(pb) + '_mean' for pb in range(6) for c in mean_columns] + \
              [c + '_all_var' for c in var_columns] + \
              [c + '_pb' + str(pb) + '_var' for pb in range(6) for c in var_columns]
all_columns
# auxiliary function for accumulating sum of values and sum of values squares by passbands
# indexes correspond to positions of fields in source datasets
def convert_row(row):
    current_id = int(row[0])
    current_pb = int(row[2])
    return current_id, current_pb, [1,                                 # count
                                    float(row[1]), float(row[1]) ** 2, # mjd
                                    float(row[3]), float(row[3]) ** 2, # flux
                                    float(row[4]), float(row[4]) ** 2, # flux_err
                                    int(row[5])]                       # detected
# auxiliary function for calculating means and vars for one object including by passbands
# rows_dict - dictionary for one object with passbands as keys and sum of convert_row's result as values
# indexes correspond to positions of fields in convert_row's output
def estimate_means_vars(rows_dict):
    values = np.array(list(rows_dict.values()))
    all_count = sum(values[:, 0])
    all_means = [
                sum(values[:, 1]) / all_count, # mjd
                sum(values[:, 3]) / all_count, # flux
                sum(values[:, 5]) / all_count, # flux_err
                sum(values[:, 7]) / all_count  # detected
    ]
    # Var(X) = E(X^2) - (E(X))^2
    all_vars = [
                sum(values[:, 2]) / all_count - all_means[0] ** 2, # mjd
                sum(values[:, 4]) / all_count - all_means[1] ** 2, # flux
                sum(values[:, 6]) / all_count - all_means[2] ** 2  # flux_err
    ]
    for pb in range(6):
        values = rows_dict[pb]
        count = values[0]
        pb_means = [
                values[1] / count,
                values[3] / count,
                values[5] / count,
                values[7] / count
        ]
        all_means += pb_means
        pb_vars = [
                values[2] / count - pb_means[0] ** 2, 
                values[4] / count - pb_means[1] ** 2, 
                values[6] / count - pb_means[2] ** 2 
        ]
        all_vars += pb_vars
        
    return [all_count] + all_means + all_vars
# main function for aggregating dataset from file to file
def aggregate_file(in_file_name, out_file_name, out_columns, verbose = -1):
    with open(in_file_name, mode = 'r') as in_file, \
         open(out_file_name, mode = 'a') as out_file:

        # open reader and skip header
        reader = csv.reader(in_file)
        next(reader, None)
    
        # open writer and write new header
        csv.register_dialect('lineterminator', lineterminator = '\n')
        writer = csv.writer(out_file, 'lineterminator')
        writer.writerow(out_columns)
    
        current_id = None
        
        # loop through rows in reader
        for i, row in enumerate(reader):
            curr_id, curr_pb, agg_row = convert_row(row)
        
            # the same object
            if current_id == curr_id:
                # passband found
                if curr_pb in current_agg_rows.keys():
                    current_agg_rows[curr_pb] = [i + j for i, j in zip(current_agg_rows[curr_pb], agg_row)]
                # new passband
                else:
                    current_agg_rows[curr_pb] = agg_row
                
            # the new object
            else:
                # save previous object if exist
                if not current_id is None:
                    # calculate estimates of mean and var for previous object
                    final_row = estimate_means_vars(current_agg_rows)
                    writer.writerow([current_id] + final_row)
                # start new object
                current_id, current_pb = curr_id, curr_pb
                current_agg_rows = {}
                current_agg_rows[current_pb] = agg_row
                
            # verbose
            if (verbose > 0) and (i % verbose == 0):
                print('Row:', i, time.ctime())
            
        # save the last object if exist
        if not current_id is None:
            # calculate estimates of mean and var for the last object
            final_row = estimate_means_vars(current_agg_rows)
            writer.writerow([current_id] + final_row)
%%time
aggregate_file('../input/training_set.csv', 'agg_training_set.csv', all_columns)
%%time
aggregate_file('../input/test_set.csv', 'agg_test_set.csv', all_columns, verbose = 100000000)
train_set = pd.read_csv('agg_training_set.csv')
train_set.head()
train_set.shape
train_set.info(null_counts = True)