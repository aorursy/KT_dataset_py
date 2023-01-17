# coding: utf-8

class GettingStarted:

    def __init__(self):

        self.data = None



    def reading_in_train_csv(self):

        """Python has a nice csv reader, which reads each line of a file into memory. You can read in each row and just append a list. From there, you can quickly turn it into an array. """



        import csv as csv 

        import numpy as np

        csv_file_object = csv.reader(open('../input/train.csv', 'r'))

        header = next(csv_file_object)



        data = []

        for row in csv_file_object:

            data.append(row)

        data = np.array(data)



        print ('print data', data)

        print ('print data[0]', data[0])

        print ('print data[-1]', data[-1])

        print ('print data[0,3]', data[0,3])

        self.data = data



    def have_data_play_with_it(self):

        pass



    @staticmethod

    def writing_gender_model():

        import csv

        test_file = open('../input/test.csv', 'r')

        test_file_object = csv.reader(test_file)

        header = test_file_object.next()

        prediction_file = open('genderbasedmodel.csv', 'wb')

        prediction_file_object = csv.writer(prediction_file)

        prediction_file_object.writerow(['PassengerId', 'Survived'])

        for row in test_file_object:

            gender = {'female': '1', 'male': '0'}

            prediction_file_object.writerow([row[0], gender[row[3]]])

        test_file.close()

        prediction_file.close()



    def second_submission(self):

        import numpy as np

        import csv

        if self.data is None:

            self.reading_in_train_csv()

        data = self.data



        fare_ceiling, fare_bracket_size, number_of_classes = 40, 10, len(np.unique(data[0:,2]))

        number_of_price_brackets = int(fare_ceiling / fare_bracket_size)

        survival_table = np.zeros((2, number_of_classes, number_of_price_brackets))

        for i in range(number_of_classes):

            for j in range(number_of_price_brackets):

                matched = (data[0:,2].astype(np.float) == i+1) & (data[0:,9].astype(np.float) >= j*fare_bracket_size) & (data[0:,9].astype(np.float) < (j+1)*fare_bracket_size)



                women_only_stats = data[(data[0:,4] == 'female') & matched, 1]

                men_only_stats = data[(data[0:,4] == 'male') & matched, 1]

                survival_table[0,i,j] = np.mean(women_only_stats.astype(np.float))

                survival_table[1,i,j] = np.mean(men_only_stats.astype(np.float))

        survival_table[survival_table != survival_table] = 0.

        print ('survival_table:', survival_table)



        #write to genderclassmodel.csv

        test_file = open('../input/test.csv', 'r')

        test_file_object = csv.reader(test_file)

        header = next(test_file_object)

        predictions_file = open('genderclassmodel.csv', 'w')

        p = csv.writer(predictions_file)

        p.writerow(['PassengerId', 'Survived'])



        for row in test_file_object:

            for j in range(number_of_classes):

                try:

                    row[8] = float(row[8])

                except ValueError:

                    bin_fare = 3 - float(row[1])

                    break



                if row[8] > fare_ceiling:

                    bin_fare = number_of_price_brackets - 1

                    break



                if (j+1) * fare_bracket_size > row[8] >= j * fare_bracket_size:

                    bin_fare = j

                    break



                p.writerow([row[0], '%d' % int(survival_table[0 if row[3] == 'female' else 1, float(row[1])-1, bin_fare])])

        test_file.close()

        predictions_file.close()



if __name__ == '__main__':

    g = GettingStarted()

    g.second_submission()

def data_munging():

    import pandas as pd

    import pylab as P

    

    df = pd.read_csv('../input/train.csv', header=0)

    #df['Age'].hist()

    #P.show()

    

    df['Age'].dropna().hist(bins=16, range=(0,80), alpha = .5)

    P.show()

data_munging()