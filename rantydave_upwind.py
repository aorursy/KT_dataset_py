%matplotlib inline

import json

import matplotlib.pyplot as plt

import numpy as np

from math import radians, sin, cos, pi
class SyntheticWind:

    def __init__(self, change, shift_amplitude, shift_period):

        self.change = change / 3600

        self.shift_amplitude = shift_amplitude

        self.shift_period = shift_period if shift_period > 60 else 60

        self.shifting_clockwise = False

        self.last_direction = 0

        

    def direction(self, time):

        dirn = self.change * time + sin((time / self.shift_period) * 2 * pi) * self.shift_amplitude

        self.shifting_clockwise = (dirn > self.last_direction)

        self.last_direction = dirn

        return dirn
class Boat:

    pointing_angle = 40

    target_speed = 8

    tacking_loss = 0.2

    

    def __init__(self, max_steps):

        self.x = 0

        self.y = 0

        self.starboard = True

        self.track_x = np.empty(max_steps+1)

        self.track_y = np.empty(max_steps+1)

        self.track_x[0] = 0

        self.track_y[0] = 0

        self.step_no = 1

        

    def step(self, tacking, wind_angle):

        speed = Boat.target_speed * (1-Boat.tacking_loss) if tacking else Boat.target_speed

        self.starboard = not self.starboard if tacking else self.starboard

        relative_heading = -Boat.pointing_angle if self.starboard else Boat.pointing_angle

        delta_x = sin(radians(wind_angle + relative_heading)) * speed * 1/60

        delta_y = cos(radians(wind_angle + relative_heading)) * speed * 1/60

        self.x += delta_x

        self.y += delta_y

        self.track_x[self.step_no] = self.x

        self.track_y[self.step_no] = self.y

        self.step_no += 1
class EnforcedTack:

    def __init__(self, time=1000000, starboard=False):

        self.time = time

        self.starboard = starboard

            

class Simulation:

    minutes_sailed = 90

    

    def this_colour():

        for c in ['r', 'g', 'b', 'k']:

            yield c



    def simulate(wind, 

                 corner_tack_time, 

                 shifts_tack_enforce=EnforcedTack(), 

                 early_shifts_tack_enforce=EnforcedTack()):

        left = Boat(Simulation.minutes_sailed)

        right = Boat(Simulation.minutes_sailed)

        shifts = Boat(Simulation.minutes_sailed)

        early_shifts = Boat(Simulation.minutes_sailed)

        wind_experienced = np.empty(Simulation.minutes_sailed)



        # Run the simulation

        for minute in range(Simulation.minutes_sailed):

            twa = wind.direction(minute*60)

            left_tack = minute == corner_tack_time

            right_tack = minute in [3, corner_tack_time]

            shift_tack = not (shifts.starboard == (twa>0))

            early_shift_tack = not (early_shifts.starboard == wind.shifting_clockwise)

            

            if minute > shifts_tack_enforce.time:

                 shift_tack = not (shifts.starboard == shifts_tack_enforce.starboard)

            if minute > early_shifts_tack_enforce.time:

                 early_shift_tack = not (early_shifts.starboard == early_shifts_tack_enforce.starboard)



            left.step(left_tack, twa)

            right.step(right_tack, twa)

            shifts.step(shift_tack, twa)

            early_shifts.step(early_shift_tack, twa)

            wind_experienced[minute] = twa



        # Graph axes (in nm)

        plt.xlim(-8, 8)

        plt.ylim(0, 12)

        

        # Draw on the laylines at the mean wind angle

        mwa = wind.change * Simulation.minutes_sailed * 60

        plt.scatter(0, 11)

        plt.plot([     4 * sin(radians(mwa-140)),  0,      4 * sin(radians(mwa+140)) ],

                 [11 + 4 * cos(radians(mwa-140)), 11, 11 + 4 * cos(radians(mwa+140)) ], '--k', linewidth=0.5)

        

        # Paint on the tracks

        colour = Simulation.this_colour()

        for sim in [left, right, shifts, early_shifts]:

            plt.plot(sim.track_x[:sim.step_no], 

                     sim.track_y[:sim.step_no], 

                     next(colour))

        plt.show()

        

        # Wind graph

        plt.plot(range(Simulation.minutes_sailed), wind_experienced, 'k')

        plt.show()
Simulation.simulate(SyntheticWind(10, 0, 0), 45, EnforcedTack(60), EnforcedTack(60))
Simulation.simulate(SyntheticWind(0, 10, 1800), 45)
Simulation.simulate(SyntheticWind(10, 10, 1800), 45, EnforcedTack(70))
Simulation.simulate(SyntheticWind(20, 10, 1800), 35, EnforcedTack(70))