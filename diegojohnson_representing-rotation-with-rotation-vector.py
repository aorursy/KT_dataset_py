import numpy as np

from math import sin, cos, atan2, sqrt

from cv2 import Rodrigues
# Euler Angles -> Rotation Matrix -> Rotation Vector

def a2v(yaw, pitch, roll):

    # Euler Angle -> Rotation Matrix

    # I think the pitch and yaw should be exchanged

    yaw, pitch = pitch, yaw

    Y = np.array([[cos(yaw), -sin(yaw), 0],

                  [sin(yaw), cos(yaw), 0],

                  [0, 0, 1]])

    P = np.array([[cos(pitch), 0, sin(pitch)],

                  [0, 1, 0],

                  [-sin(pitch), 0, cos(pitch)]])

    R = np.array([[1, 0, 0],

                  [0, cos(roll), -sin(roll)],

                  [0, sin(roll), cos(roll)]])

    rotation_m = np.dot(Y, np.dot(P, R))

    

    

    # Rotation Matrix -> Rotation Vector

    rotation_v = Rodrigues(rotation_m)[0]

    rotation_v = np.squeeze(rotation_v)

    

    return rotation_v
print(a2v(0.162877, 0.00519276, -3.02676))
# Rotation Vector -> Rotation Matrix -> Euler Angles

def v2a(rotation_v):

    # Rotation Vector -> Rotation Matrix

    R = Rodrigues(rotation_v)[0]

    

    sq = sqrt(R[0,0] ** 2 +  R[1,0] ** 2)



    if  not (sq < 1e-6) :

        roll = atan2(R[2,1] , R[2,2])

        yaw = atan2(-R[2,0], sq)

        pitch = atan2(R[1,0], R[0,0])

    else :

        roll = atan2(-R[1,2], R[1,1])

        yaw = atan2(-R[2,0], sq)

        pitch = 0



    return yaw, pitch, roll
v = np.array([-3.01748673, 0.00632166, 0.24673163])

print(v2a(v))