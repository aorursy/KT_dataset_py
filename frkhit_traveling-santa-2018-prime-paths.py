import math

import numpy as np
import pandas as pd


class KaggleUtils(object):

    def __init__(self):
        self.file_cities = "../input/cities.csv"
        self.file_submission = "submission.csv"
        self.prime_set = set()
        self.city_id_array = None
        self.loc_array = None

    def load_cities(self, ):
        cities = pd.read_csv(self.file_cities)
        city_id_array = cities.CityId.astype(np.int32)
        loc_array = np.vstack([cities.X.astype(np.float32), cities.Y.astype(np.float32)]).transpose()
        self.init_prime_set(len(city_id_array))
        self.city_id_array = city_id_array
        self.loc_array = loc_array
        return city_id_array, loc_array

    def init_prime_set(self, n):
        if self.prime_set:
            return

        def is_prime(int_num) -> bool:
            for j in range(2, int(math.sqrt(int_num)) + 1):
                if int_num % j == 0:
                    return False

            return True

        for i in range(n):
            if is_prime(i + 1):
                self.prime_set.add(i + 1)

    def load_tour(self, tour_file=None) -> np.ndarray:
        if tour_file is None:
            tour_file = self.file_submission
        tour = pd.read_csv(tour_file)
        tour = tour.Path.values.astype(np.int32)
        return tour

    def save_tour(self, tour: np.ndarray):
        # check tour:
        # 1) include all city;
        # 2) forward-tour is better than backward-tour
        tour, distance = self.check_tour(tour)

        # save tour in file
        with open(self.file_submission, "w") as f:
            f.write("Path\n")
            for i in tour:
                f.write(str(i))
                f.write("\n")
            print("success to save tour in {}".format(self.file_submission))

        # calc distance
        print("total distance is {}".format(distance))

    def check_tour(self, tour: np.ndarray) -> (np.ndarray, float):
        # check if all city_id included in tour
        city_id_set = set([city_id for city_id in self.city_id_array])
        city_id_set_in_tour = set([city_id for city_id in tour])
        assert city_id_set == city_id_set_in_tour

        # calc distance
        forward_distance = self.calc_distance(tour)
        backward_tour = tour[::-1]
        backward_distance = self.calc_distance(backward_tour)
        if forward_distance < backward_distance:
            return tour, forward_distance
        else:
            print("backward tour is better than forward tour, use backward tour instead!")
            return backward_tour, backward_distance

    def calc_distance(self, tour: np.ndarray) -> float:
        def cal_euc_distance(loc_i, loc_j):
            return math.sqrt(math.pow(loc_i[0] - loc_j[0], 2) + math.pow(loc_i[1] - loc_j[1], 2))

        dist = 0.0
        for i in range(1, tour.shape[0]):
            city_id = tour[i]
            pre_city_id = tour[i - 1]
            d = cal_euc_distance(self.loc_array[pre_city_id], self.loc_array[city_id])
            if i % 10 == 0 and city_id in self.prime_set:
                d *= 1.1
            dist += d
        return dist
