results_of_solutions = []
def round_by(value, decimal_places):

    return int(value * (10 ** decimal_places) + 0.5) / (10.0 ** decimal_places)
class result:

    name = ""

    results = []

    needed_time = 0

    

    def __init__(self, name, results, needed_time):

        self.name = name

        self.results = results

        self.needed_time = needed_time
def calculate_results(solution, name):

    start_time = time.perf_counter()

    results = []

    for data in test_data:

        results.append(round_by(solution.get_guess(data), 1))

    needed_time = time.perf_counter() - start_time

    show_results(results)

    results_of_solutions.append(result(name, results, needed_time))
from prettytable import PrettyTable

import time  # import not needed here but already put here so that I don't need to care about it later



class Final_result:

    header = []

    result = []

    

    def __init__(self, header):

        self.header = header

        

    def add_entry(self, name_of_solution, results_of_solution, needed_time):

        output = [name_of_solution]

        for result in results_of_solution:

            output.append(result)

        time_in_ms = needed_time * 1000 * 1000

        rounded_time = round_by(time_in_ms, 2)

        output.append("%s µs" % rounded_time)

        self.result.append(output)

    

    def add_entries(self, results):

        for result in results:

            self.add_entry(result.name, result.results, result.needed_time)

    

    def change_header(self, header):

        self.header = header

        

    def print_as_table(self):

        pretty_table = PrettyTable(field_names=self.header)

        for row in self.result:

            pretty_table.add_row(row)

        print(pretty_table.get_string())
# given_data[which_set_of_data][0 = TV_time; 1 = deep_sleep_time]

given_data = [(0.3, 5.8), (2.2, 4.4), (0.5, 6.5), (0.7, 5.8), (1.0, 5.6), (1.8, 5.0), (3.0, 4.8), (0.2, 6.0), (2.3, 6.1)]

TV = 0

DEEP_SLEEP = 1



test_data = [value * 0.5 + 0.5 for value in range(8)]  # all values from 0.5 to 4.0 in steps of 0.5

print(test_data)



given_data = sorted(given_data, key=lambda x: x[TV])  # sort the list by time that is spend watching TV

print(given_data)
import matplotlib.pyplot as plt



tv_times_of_given_data = [data[TV] for data in given_data]  # all TV data as List

deep_sleep_of_given_data = [data[DEEP_SLEEP] for data in given_data]  # all DEEP_SLEEP data as list



def show_results(tv_times):

    plt.scatter(tv_times_of_given_data, deep_sleep_of_given_data)

    if tv_times != []:

        plt.scatter(test_data, tv_times)

    plt.show()

    

# only displays the known data

def show_given_data():

    show_results([])
show_given_data()
class value_guesser:

    def __init__(self):

        raise NotImplementedError

    

    def get_guess(self, tv_time):

        raise NotImplementedError
class closest_left_and_right(value_guesser):

    def __init__(self):

        pass

    

    def get_guess(self, tv_time):

        first_value_left_and_right = self.get_first_value_left_and_right(tv_time)  # one value per side

        avg_value = (first_value_left_and_right[0] + first_value_left_and_right[1]) / 2

        avg_value = avg_value

        return avg_value

    

    def get_first_value_left_and_right(self, tv_time):

        # might be changed later

        smaller = 0

        bigger = 0

        while smaller < len(given_data) and given_data[smaller][TV] < tv_time:

            smaller += 1

        if smaller == 0:

            bigger = 0 if given_data[smaller][TV] else 1

        else:

            smaller -= 1

            if smaller >= len(given_data) - 1:

                smaller = len(given_data) - 1

                bigger = smaller

            else:

                bigger = smaller + 1

        return [given_data[smaller][DEEP_SLEEP], given_data[bigger][DEEP_SLEEP]]
calculate_results(closest_left_and_right(), "closest left and right")
class closest_dot(value_guesser):

    def __init__(self):

        pass

        

    def get_guess(self, value):

        return given_data[self.find_closest(value)][DEEP_SLEEP]

    

    def find_closest(self, value):

        i = 0

        while i < len(given_data) and given_data[i][TV] < value:

            i += 1

        return i - 1 if i == len(given_data) else i
calculate_results(closest_dot(), "closest dot")
class average_of_all_dots(value_guesser):

    average = 0

    def __init__(self):

        self.average = sum([data[DEEP_SLEEP] for data in given_data])

        self.average /= len(given_data)

        

    def get_guess(self, value):

        return self.average
calculate_results(average_of_all_dots(), "average of known dots")
class linear_of_two_closest(value_guesser):

    def __init__(self):

        pass

        

    def get_guess(self, value):

        (dot1, dot2) = self.get_two_closest_dots(value)

        m = (dot2[DEEP_SLEEP] - dot1[DEEP_SLEEP]) / (dot2[TV] - dot1[TV])

        b = dot1[DEEP_SLEEP] - m * dot1[TV]

        return m * value + b

    

    def get_two_closest_dots(self, value):

        indexes_of_the_closest_two = [0, 1]

        i = 2

        indexes_of_the_closest_two = self.sort(indexes_of_the_closest_two, value)

        while i < len(given_data):

            if abs(given_data[i][TV] - value) < abs(given_data[indexes_of_the_closest_two[1]][TV] - value):

                indexes_of_the_closest_two[1] = i

                indexes_of_the_closest_two = self.sort(indexes_of_the_closest_two, value)

            i += 1

        return [given_data[indexes_of_the_closest_two[0]], given_data[indexes_of_the_closest_two[1]]]

            

    def sort(self, indexes, value):

        return sorted(indexes, key=lambda x: abs(given_data[x][TV] - value))
calculate_results(linear_of_two_closest(), "linear two closest dots")
class linear_regression(value_guesser):

    beta = 0

    alpha = 0

    def __init__(self):

        avg_x = sum([data[TV] for data in given_data]) / len(given_data)

        avg_y = sum([data[DEEP_SLEEP] for data in given_data]) / len(given_data)

        for data in given_data:

            self.beta += (data[TV] - avg_x) * (data[DEEP_SLEEP] - avg_y)

        self.beta /= len(given_data)

        self.alpha = avg_y - self.beta * avg_x

    

    def get_guess(self, value):

        return self.beta * value + self.alpha
calculate_results(linear_regression(), "linear regression")
table_header = ["name of solution"]

for data in test_data:

    table_header.append(data)

table_header.append("time")

result_table = Final_result(table_header)

result_table.add_entries(results_of_solutions)

result_table.print_as_table()