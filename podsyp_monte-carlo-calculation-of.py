import random
points_n = 100000000

length = 10000
class Monte_carlo_for_pi():

    n_points = None

    length = None

    

    def __init__(self, n_points=100000, length=1000):

        ''' Input: 

            n_points - number of points for experiment, 

            length - side length of a square'''

        self.n_points = n_points

        self.length = n_points

        self.center_x = self.length / 2

        self.center_y = self.length / 2

        self.radius = self.length / 2

    

    def in_circle(self, x, y):

        return (x - self.center_x)**2 + (y - self.center_y)**2 < self.radius**2

    

    def compute_pi(self):

        is_inside = 0

        for i in range(self.n_points): 

            if self.in_circle(random.randint(1, self.length), random.randint(1, self.length)):

                is_inside += 1

        return (is_inside / self.n_points) * 4
%%time

mc = Monte_carlo_for_pi(n_points=points_n, length=length)

mc.compute_pi()