import csv
from datetime import datetime


def clean_data(input_data_path='../input/train.csv', output_data_path='../data/train_cleaned.csv'):
    """
    Clean the data set, removing any row with missing values,
    delimiter longitudes and latitudes to fit only NY city values,
    only fare amount greater than 0,
    and passenger count greater than 0 and lesser than 7,
    i also removed the header as i'm using tensorflow to load data.
    :param input_data_path: path containing the raw data set.
    :param output_data_path: path to write the cleaned data.
    """
    with open(input_data_path, 'r') as inp, open(output_data_path, 'w', newline='') as out:
        writer = csv.writer(out)
        count = 0
        for row in csv.reader(inp):
            # Remove header
            if count > 0:
                # Only rows with non-null values
                if len(row) == 8:
                    try:
                        fare_amount = float(row[1])
                        pickup_longitude = float(row[3])
                        pickup_latitude = float(row[4])
                        dropoff_longitude = float(row[5])
                        dropoff_latitude = float(row[6])
                        passenger_count = float(row[7])
                        if ((-76 <= pickup_longitude <= -72) and (-76 <= dropoff_longitude <= -72) and
                                (38 <= pickup_latitude <= 42) and (38 <= dropoff_latitude <= 42) and
                                (1 <= passenger_count <= 6) and fare_amount > 0):
                            writer.writerow(row)
                    except:
                        pass
            count += 1


def pre_process_train_data(input_data_path='data/train_cleaned.csv', output_data_path='data/train_processed.csv'):
    """
    Pre process the train data, deriving, year, month, day and hour for each row.
    :param input_data_path: path containing the full data set.
    :param output_data_path: path to write the pre processed set.
    """
    with open(input_data_path, 'r') as inp, open(output_data_path, 'w', newline='') as out:
        writer = csv.writer(out)
        for row in csv.reader(inp):
            pickup_datetime = datetime.strptime(row[2], '%Y-%m-%d %H:%M:%S %Z')
            row.append(pickup_datetime.year)
            row.append(pickup_datetime.month)
            row.append(pickup_datetime.day)
            row.append(pickup_datetime.hour)
            row.append(pickup_datetime.weekday())
            writer.writerow(row)


def pre_process_test_data(input_data_path='data/test.csv', output_data_path='data/test_processed.csv'):
    """
    Pre process the test data, deriving, year, month, day and hour for each row.
    :param input_data_path: path containing the full data set.
    :param output_data_path: path to write the pre processed set.
    """
    with open(input_data_path, 'r') as inp, open(output_data_path, 'w', newline='') as out:
        writer = csv.writer(out)
        count = 0
        for row in csv.reader(inp):
            if count > 0:
                pickup_datetime = datetime.strptime(row[1], '%Y-%m-%d %H:%M:%S %Z')
                row.append(pickup_datetime.year)
                row.append(pickup_datetime.month)
                row.append(pickup_datetime.day)
                row.append(pickup_datetime.hour)
                row.append(pickup_datetime.weekday())
                writer.writerow(row)
            else:
                # Only the header
                writer.writerow(row)
            count += 1


def split_data(input_data_path, train_data_path, validation_data_path, ratio=30):
    """
    Splits the csv file (meant to generate train and validation sets).
    :param input_data_path: path containing the full data set.
    :param train_data_path: path to write the train set.
    :param validation_data_path: path to write the validation set.
    :param ratio: ration to split train and validation sets, (default: 1 of every 30 rows will be validation or 0,033%)
    """
    with open(input_data_path, 'r') as inp, open(train_data_path, 'w', newline='') as out1, \
            open(validation_data_path, 'w', newline='') as out2:
        writer1 = csv.writer(out1)
        writer2 = csv.writer(out2)
        count = 0
        for row in csv.reader(inp):
            if count % ratio == 0:
                writer2.writerow(row)
            else:
                writer1.writerow(row)
            count += 1
