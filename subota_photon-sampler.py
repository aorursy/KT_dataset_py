import numpy as np

import scipy.stats as st

import matplotlib.pyplot as plt

import pandas as pd

from collections import defaultdict

import time

import unittest

t = unittest.TestCase()



SPACE_DIMENSIONS = 2
class Points(np.ndarray):

    '''ndarray sized (SPACE_DIMENSIONS,...) with named coordinates x,y'''

    @staticmethod

    def of(coords):

        p = np.asarray(coords).view(Points)

        assert p.shape[0] == SPACE_DIMENSIONS

        return p

    @property

    def x(self): return self[0]

    @property

    def y(self): return self[1]



class Lines(np.ndarray):

    '''ndarray shaped (3,...) with named line parameters a,b,c'''

    @staticmethod

    def of(abc):

        lp = np.asarray(abc).view(Lines)

        assert lp.shape[0] == 3

        return lp

    @property

    def a(self): return self[0]

    @property

    def b(self): return self[1]

    @property

    def c(self): return self[2]



    def intersections(self, hyperplanes) -> Points:

        '''

        https://stackoverflow.com/a/20679579/2082707

        answered Dec 19 '13 at 10:46 by rook

        Adapted for numpy matrix operations by Subota



        Intersection points of lines from the first set with hyperplanes from the second set.

        Currently only 2D sapce supported, e.g. the second lanes is lines, too.

        @hyperplanes parametrical equation coeffs. For 2D it is also Lines

        @return array of intersection coordinates as Points, sized:

            - SPACE_DIMENSIONS for intersection coordinates

            - n1 for the number of lines passed in L1

            - n2 for the number of lines passed in L2

        '''

        l1 = np.reshape(self, (*self.shape,1))

        l2 = hyperplanes



        d  = l1.a * l2.b - l1.b * l2.a

        dx = l1.c * l2.b - l1.b * l2.c

        dy = l1.a * l2.c - l1.c * l2.a

        d[d==0.] = np.nan

        x = dx / d

        y = dy / d

        return Points.of((x,y))



class LineSegments(np.ndarray):

    '''Wrapper around ndarray((2,SPACE_DIMENSIONS)) to access endPoint1, endPoint2 and coordinates x,y by names'''

    @staticmethod

    def of(point_coords):

        ls = np.asarray(point_coords).view(LineSegments)

        assert ls.shape[0] == 2

        assert ls.shape[1] == SPACE_DIMENSIONS

        return ls



    @property

    def endPoint1(self): return Points.of(self[0])

    @property

    def endPoint2(self): return Points.of(self[1])



    @property

    def x(self): return self[:,0]

    @property

    def y(self): return self[:,1]



    def length(self) -> np.array:

        dif = self.endPoint1 - self.endPoint2

        return np.sqrt(dif.x*dif.x + dif.y*dif.y).view(np.ndarray)



    def lines(self) -> Lines:

        '''

        https://stackoverflow.com/a/20679579/2082707

        answered Dec 19 '13 at 10:46 by rook

        Adapted for numpy matrix operations by Subota



        Calculates the line equation Ay + Bx - C = 0, given two points on a line.

        Horizontal and vertical lines are Ok

        @return returns an array of Lines parameters sized:

            - 3 for the parameters A, B, and C

            - n for the number of lines calculated

        '''

        p1, p2 = self.endPoint1, self.endPoint2

        a = (p1.y - p2.y)

        b = (p2.x - p1.x)

        c = - ( p1.x*p2.y - p2.x*p1.y)

        return Lines.of((a, b, c))



    def intersections(self, other) -> Points:

        '''

        Returns intersection points for line sets,

        along with the true/false matrix for do intersections lie within the segments or not.

        @other LineSegments to find intersections with. Sized:

            - 2 for the endPoint1 and endPoint2

            - SPACE_DIMENSIONS

            - n1 for the number of segments in the first set

            Generally speaking these must be hyper-planes in N-dimensional space

        @return a tuple with two elements

            0. boolean matrix sized(n1,n2), True the intersection to fall within the segments, False otherwise.

            1. intersection Points sized (SPACE_DIMENSIONS, n1, n2)

        '''

        s1, s2 = self, other

        l1, l2 = self.lines(), other.lines()

        il = l1.intersections(l2)

        s1 = s1.reshape((2,SPACE_DIMENSIONS,-1,1))

        s1p1, s1p2 = s1.endPoint1, s1.endPoint2

        s2p1, s2p2 = s2.endPoint1, s2.endPoint2



        # Allowance for intersection point ocation in case, helps for strictly vertical and horizontal lines

        ROUNDING_THRESHOLD = np.array(1e-10)

        which_intersect = (

            # intersection point is within the first interval

            (il.x <= np.maximum(s1p1.x, s1p2.x) + ROUNDING_THRESHOLD) &

            (il.x >= np.minimum(s1p1.x, s1p2.x) - ROUNDING_THRESHOLD) &

            (il.y <= np.maximum(s1p1.y, s1p2.y) + ROUNDING_THRESHOLD) &

            (il.y >= np.minimum(s1p1.y, s1p2.y) - ROUNDING_THRESHOLD) &



            # intersection point is within the second interval

            (il.x <= np.maximum(s2p1.x, s2p2.x) + ROUNDING_THRESHOLD) &

            (il.x >= np.minimum(s2p1.x, s2p2.x) - ROUNDING_THRESHOLD) &

            (il.y <= np.maximum(s2p1.y, s2p2.y) + ROUNDING_THRESHOLD) &

            (il.y >= np.minimum(s2p1.y, s2p2.y) - ROUNDING_THRESHOLD)

        )

        return which_intersect, il



# diagonal, vertical, horizontal

t.assertTrue( np.allclose( LineSegments.of([[[-1.],[-1]],[[1],[1]]]).lines().flat, np.array([-2,2,0])))

t.assertTrue( np.allclose( LineSegments.of([[[0.],[-1]],[[0],[1]]]). lines().flat,  np.array([-2,0,0])))

t.assertTrue( np.allclose( LineSegments.of([[[3.],[1]],[[-4],[1]]]). lines().flat,  np.array([0,-7,-7])))

t.assertEqual( LineSegments.of([Points.of([0,0]),Points.of([3,4])]).length(), 5)
def demo_intersect_lines():

    seg1 = LineSegments.of( st.uniform.rvs(size=(2,SPACE_DIMENSIONS, 2), random_state=19)   )

    seg2 = LineSegments.of( st.uniform.rvs(size=(2,SPACE_DIMENSIONS, 3), random_state=15)+1 )

    l1, l2 = seg1.lines(), seg2.lines()

    i = l1.intersections(l2)



    plt.plot(seg1.x, seg1.y, '-', c='green')

    plt.plot(seg2.x, seg2.y, '-', c='blue')



    plt.plot(i.x, i.y, '+', c='red', markersize=20)

    plt.title('Extended Line Intersections')



    plt.axis('off')



def demo_intersect_segments():

    seg1 = LineSegments.of( st.uniform.rvs(size=(2,SPACE_DIMENSIONS, 4), random_state=1) )

    seg2 = LineSegments.of( st.uniform.rvs(size=(2,SPACE_DIMENSIONS, 5), random_state=2) )

    plt.plot(seg1.x, seg1.y, '-', c='black')

    plt.plot(seg2.x, seg2.y, '-', c='lightgrey')

    w, i = seg1.intersections(seg2)

    plt.plot(i.x[w], i.y[w], '+', c='red', markersize=20)

    plt.title('Segment Intersections')

    plt.axis('off')



f, ax = plt.subplots(ncols=2)

f.set_size_inches(12,4)

plt.sca(ax[0])

demo_intersect_lines()

plt.sca(ax[1])

demo_intersect_segments()
SEGMENT_ENDPOINTS = 2

NUM_WALLS = 7

SOURCE = Points.of( (0.,0.) )

DETECTOR = LineSegments.of( ((8.,-1), (8.,+1)) )



walls = LineSegments.of( np.zeros((SEGMENT_ENDPOINTS,SPACE_DIMENSIONS, NUM_WALLS)) )

SLIT_WIDTH, SLITS_APART = 0.05, 0.5

# The wall with slits

# above the slits

walls[:,:,1] = ( (6.,+1.), (6.,+SLITS_APART/2+SLIT_WIDTH) )

# between the slits

walls[:,:,2] = ( (6.,-SLITS_APART/2), (6.,+SLITS_APART/2) )

# below the slits

walls[:,:,3] = ( (6.,-1.), (6.,-SLITS_APART/2-SLIT_WIDTH) )

# square box

walls[:,:,4] = ( (-1,-1), (-1,+1)) # left wall

walls[:,:,5] = ( (-1,+1), (+8.1,+1)) # top

walls[:,:,6] = ( (+8.1,+1), (+8.1,-1)) # right

walls[:,:,0] = ( (+8.1,-1), (-1,-1)) # bottom





def plot_experimet_setup(walls, detector, source):

    plt.plot(*source,'o', color='red', label='Source')



    wall_lines = plt.plot(walls.x, walls.y, '-', c='black', linewidth=1);

    wall_lines[1].set_label('Walls')



    plt.plot(detector.x, detector.y, '-', c='green', linewidth=4, label='Detector');



    plt.gcf().set_size_inches(12,5)

    plt.legend(loc = 'upper center');

plot_experimet_setup(walls, DETECTOR, SOURCE)
detections = []

np.random.seed(1254785)



MIN_STEPS_TO_DETECTION = 2

BATCH_SIZE = 50_000



def shifter_uniform_destination(r0: Points):

    '''Shift is so that a photon arrives to a uniformly picked location in the test setup area, regardkess current position'''

    target_x = st.uniform(loc=-1, scale = 1+6.).rvs(r0.shape[1])

    target_y = st.uniform(loc=-1, scale = 1+1.).rvs(r0.shape[1])

    return np.array([target_x, target_y]) - r0



# Start with a bunch of photons all at the SOURCE (0,0) point

photons = Points.of( np.zeros((SPACE_DIMENSIONS,BATCH_SIZE)) )

lengths = np.zeros(BATCH_SIZE)

steps = np.zeros(BATCH_SIZE, dtype='B')



start = time.monotonic()

last_reported = len(detections)

epoch = 0

while len(detections)<1_000_000:

    epoch += 1

    if last_reported <= len(detections) - 50_000:

        last_reported = round(len(detections)/1000) * 1000

        print(len(detections),end=', ')

        

    # Increment each path steps counter

    steps += 1



    # Move each photon to a uniformly picked location within the experimanetal setup box

    # If the step counter >= MIN_STEPS_TO_DETECTION then move into a random position in the detection area instead

    randomInBox = Points.of( st.uniform().rvs(photons.shape) )

    randomInBox.x[...] *= 9; randomInBox.x[...] -= 1

    randomInBox.y[...] *= 2; randomInBox.y[...] -= 1

    

    randomInDetector = Points.of( st.uniform().rvs(photons.shape) )

    randomInDetector.x[...] = 8

    randomInDetector.y[...] *= 2; randomInDetector.y[...] -= 1

    

    newLoc = np.where(steps < MIN_STEPS_TO_DETECTION, randomInBox, randomInDetector)



    # Increment path length by the euclidean length of the shift

    moves = LineSegments.of( (photons, newLoc) )

    lengths += moves.length()

    photons = moves.endPoint2



    # Detect wall collisions and reset colliders to the SOURCE, reset path length and step count to zero

    colliders, _ = moves.intersections( walls )

    colliders = np.logical_or.reduce(colliders, axis=1)

    

    photons[:,colliders] = 0

    steps[colliders] = 0

    lengths[colliders] = 0



    # If still step counter >= MIN_STEPS_TO_DETECTION, register detection and reset to SOURCE

    detected = (steps >= MIN_STEPS_TO_DETECTION)

    for i in np.where(detected)[0]:

        detections += [(*photons[:,i], lengths[i])]

    photons[:,detected] = 0

    steps[detected] = 0

    lengths[detected] = 0



print('Time total: %.1f sec' % (time.monotonic()-start) )
ROUND_TO=0.01

WAVE_LENGTH = 0.08

det_array = np.array(detections).T



path_lengths = det_array[2]

detection_points = Points.of(det_array[:SPACE_DIMENSIONS])

df = pd.DataFrame({

    'x': np.round(detection_points.x/ROUND_TO)*ROUND_TO,

    'y': np.round(detection_points.y/ROUND_TO)*ROUND_TO,

    'l_sin': np.sin( 2.*np.pi / WAVE_LENGTH * path_lengths )

})

df['l_cos'] = np.sqrt(1. - df['l_sin']**2)

df = df.groupby(['x','y'], as_index=False).sum()

df['intensity'] = np.sqrt( df.l_sin**2 + df.l_cos**2 )
plt.fill_betweenx(df.y, df.x-df.intensity/np.sum(df.intensity)*2/ROUND_TO, 8, label='Intensity Sim', color='pink')

plt.hlines(y=0,xmin=0, xmax=8, linestyle=':', color='black')

plt.hlines(y=np.arange(-1,1,.5),xmin=8, xmax=8.25, linestyle='-', color='black')

plt.hlines(y=np.arange(-1,1,.1),xmin=8, xmax=8.15, linestyle='-', color='black')

#plt.hlines(y=np.arange(0.,1.,WAVE_LENGTH), xmin=-1, xmax=np.mean(df.x), linestyles=':', color='grey', label='Wave Length')



xdt = np.linspace(-1.,1.,200)

dSlit1up  = np.sqrt(((0+SLITS_APART/2. + SLIT_WIDTH) - xdt)**2 + 2**2) * 2*np.pi/WAVE_LENGTH

dSlit1dwn = np.sqrt(((0+SLITS_APART/2. - SLIT_WIDTH) - xdt)**2 + 2**2) * 2*np.pi/WAVE_LENGTH

dSlit2up  = np.sqrt(((0-SLITS_APART/2. + SLIT_WIDTH) - xdt)**2 + 2**2) * 2*np.pi/WAVE_LENGTH

dSlit2dwn = np.sqrt(((0-SLITS_APART/2. - SLIT_WIDTH) - xdt)**2 + 2**2) * 2*np.pi/WAVE_LENGTH



dAll = np.array([dSlit1up, dSlit1dwn, dSlit2up, dSlit2dwn])

theoretical = np.sqrt( np.sum(np.sin(dAll),axis=0)**2 + np.sum(np.cos(dAll),axis=0)**2 )

plt.plot(-theoretical/np.sum(theoretical)*200 + 8, xdt, ':', color='blue', label='(?) Theoretical', linewidth=1)

plot_experimet_setup(walls, DETECTOR, SOURCE)

plt.show()

plt.close('all')