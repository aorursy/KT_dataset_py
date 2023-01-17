import csv



with open("../input/input-data-task-1/test_input_data_sample.txt") as sample_input_data:

    reader = csv.DictReader(sample_input_data, dialect="excel-tab")

    for row in reader:

        #print(row)

        print(row['statement']+ ' | ' +row['section']+ ' | ' +row['citations'])
import csv



with open("../input/output-data-task-1/en_predictions_sections.tsv") as prediction_file:

    reader = csv.DictReader(prediction_file, dialect="excel-tab")

    for row in reader:

        ##print(row)

        print(row['Text'] + ' | ' + row['Prediction'] + ' | ' + row['Citation'] + "\n")