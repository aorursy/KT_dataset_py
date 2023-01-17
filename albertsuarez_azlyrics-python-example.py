import csv

import glob





def _get_csv_file_list(unzipping_output_folder):

    """

    Extract all the csv file paths given the folder path.

    :param unzipping_output_folder: Folder path.

    :return: List of CSV file paths.

    """

    csv_file_list = [i for i in glob.glob(f'{unzipping_output_folder}/**/*.csv')]

    print(f'{len(csv_file_list)} CSV files extracted.')

    return csv_file_list





def _read(csv_file_list):

    """

    Read the list of data CSVs.

    :param csv_file_list: List of CSV file paths where the data is.

    :return: CSVs read.

    """

    total = 0

    for csv_file_name in csv_file_list:

        with open(csv_file_name, 'r') as csv_file:

            rows = [row for row in csv.reader(csv_file) if row][1:]

            total += len(rows)

            print(f'{csv_file_name}: {len(rows)} songs.')

    print(f'Total: {total}')

            

            

data_folder = '/kaggle/input/azlyrics'

csv_file_list = _get_csv_file_list(data_folder)

_read(csv_file_list)
