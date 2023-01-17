import csv

import numpy as np
class PricePredicted:

    def __init__(self, data=None):

        """ Create a blank list data"""

        

        if data is None:

            data = []

        self.data = data       # create a base data



    def load_data(self):

        """ Load file AB_NYC_2019.csv"""

        

        with open('../input/nycairbnb/AB_NYC_2019.csv', encoding='utf8') as csv_file:

            csv_reader = csv.reader(csv_file)

            next(csv_reader)

            for row in csv_reader:

                self.data.append(row)

                

            csv_file.close()

            self.data = self.data[:4999]      # Read only 5000 rows

        return self.data



    def cleaning_data(self):

        """Set missing data to '0'"""

        

        self.data[self.data == ''] = '0'

        return self.data



    def convert_day_tovalue(self):

        """ We can not use the datetime type to calculate

            I change the date to 3 type:0 1 2

            0: no data

            1: before '2019-01-01'

            2: after '2019-01-01'

        """

        a = np.array(self.data[:, 12], dtype='datetime64')

        with np.nditer(a, op_flags=['readwrite']) as it:

            for x in it:

                x[np.datetime64(x) < np.datetime64('2019-01-01')] = '1'

                x[np.datetime64(x) >= np.datetime64('2019-01-01')] = '2'

        a = a.astype(str)

        a[a == '0002-01-01'] = '2'

        a[a == '0001-01-01'] = '1'

        a[a == 'NaT'] = '0'

        self.data[:, 12] = a

        return self.data





    def convert_to_array(self):

        """Convert list to array for quick calculate"""

        

        self.data = np.array(self.data)

        return self.data



    def linear_sum(self, S, n):

        """ Linear sum one algorithm on lecture class"""

        

        if n == 0:

            return 0

        else:

            return self.linear_sum(S, n - 1) + S[n - 1]



    def maxi(self, arr):

        """ Maxi one algorithm on lecture class"""

        

        currentMax = arr[0]

        n = len(arr)

        for i in range(1, n-1):

            if arr[i] > currentMax:

                currentMax = arr[i]

        return currentMax



    def get_column(self, column=0):

        """ Get a specific column"""

        

        data = self.data[:, column]

        return data

    

    def get_Data(self, column=0, value=''):

        """Get the value by the specific value

        """

        

        data_return = []

        data = np.array(np.where(self.data[:, column] == value))    # find the index of value

        for each in data:



            return "{}\n".format(self.data[each, :])





    def normalization(self, column=0):

        """ Normalization method

            I do not convert it to (0,1) because the price can be raised

            I converted it to (0, 2) format

        """

        

        a = self.data[:, column].astype(float)                                        # choose a column

        max_value = self.maxi(a)                                                      # find the max value

        min_value = np.amin(a)                                                        # find the min value

        self.data[:, column] = np.true_divide(a-min_value, max_value-min_value)*2     # convert (0, 2)

        return self.data[:, column].astype(float)

    

    def get_coefficient(self):

        """ I choose 6 figures to calculate prrice

            Each figure has unique coefficient that i choice

            *1 : minimum nights

            *2 : number of reviews

            *3 : host count

            *4 : lastest review

            *5 : preview per month

            *6 : available 365

        """

        

        number_reviews = self.normalization(11)*2            # number review

        preview_per_month = self.normalization(13)*5         # review per month

        minimum_nights = self.normalization(10)              # minimum nights

        a365 = self.normalization(15)*6                      # available 365

        lastest_review = self.normalization(12)*4            # lastest review

        host_count = self.normalization(14)*3                # host_count

        

        list_element = [number_reviews, preview_per_month, minimum_nights, a365, lastest_review, host_count]

        total = self.linear_sum(list_element, len(list_element))/21   # using linear sum algorithm

        return total

    

    def price_predicted(self):

        """ Return the price is predicted"""

        

        new_price = ['Price']

        ids = ['Id']

        prices = self.data[:, 9].astype(float)

        predicted_price = prices * self.get_coefficient()

        

        for each in predicted_price:

            each = round(each, 2)

            new_price.append(each)

        id = self.data[:, 0]

        for each in id:

            ids.append(each)

        data = ''

        for i, j in zip(ids, new_price):

            data += '{} - {}\n'.format(i, j)

        return data





    def __repr__(self):

        """ For fun(and test)"""

        

        data = ''.format(self.data[:, 2])



        return data

def run_test():

    data = PricePredicted()

    data.load_data()

    data.convert_to_array()

    data.convert_day_tovalue()

    data.cleaning_data()



    print('Who is the most expensive host in NYC?')

    a = data.get_column(9)

    max_price = data.maxi(a).astype(str)

    print("Most expensive:{}".format(max_price))

    id_1 = data.get_Data(9, max_price)

    print(id_1)

    

    print('')

    

    print('Who is the most available host?')

    b = data.get_column(14)

    most_host = data.maxi(b).astype(str)

    print("Most hosts:{}".format(most_host))

    id_2 = data.get_Data(14, most_host)

    print(id_2)

    

    print('')

    

    print('Who has the most number of reviews?')

    c = data.get_column(11)

    most_review = data.maxi(c).astype(str)

    print("Most reviews:{}".format(most_review))

    id_3 = data.get_Data(11, most_review)

    print(id_3)



    print('Ids with price predicted')

    coefficient = data.get_coefficient()

    print(coefficient)

    predicted_price = data.price_predicted()

    print(predicted_price)

run_test()






