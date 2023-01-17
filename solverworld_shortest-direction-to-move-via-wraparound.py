from kaggle_environments.envs.halite.helpers import *



def dirs_to(p1, p2, size=21):

  #Get the actions you should take to go from Point p1 to Point p2

  #using shortest direction by wraparound

  #Args: p1: from Point

  #      p2: to Point

  #      size:  size of board

  #returns: list of directions, tuple (deltaX,deltaY)

  #The list is of length 1 or 2 giving possible directions to go, e.g.

  #to go North-East, it would return [ShipAction.NORTH, ShipAction.EAST], because

  #you could use either of those first to go North-East.

  #[None] is returned if p1==p2 and there is no need to move at all

  deltaX, deltaY=p2 - p1

  if abs(deltaX)>size/2:

    #we wrap around

    if deltaX<0:

      deltaX+=size

    elif deltaX>0:

      deltaX-=size

  if abs(deltaY)>size/2:

    #we wrap around

    if deltaY<0:

      deltaY+=size

    elif deltaY>0:

      deltaY-=size

  #the delta is (deltaX,deltaY)

  ret=[]

  if deltaX>0:

    ret.append(ShipAction.EAST)

  if deltaX<0:

    ret.append(ShipAction.WEST)

  if deltaY>0:

    ret.append(ShipAction.NORTH)

  if deltaY<0:

    ret.append(ShipAction.SOUTH)

  if len(ret)==0:

    ret=[None]  # do not need to move at all

  return ret, (deltaX,deltaY)

  

      

#test it

x1=dirs_to(Point(2,2), Point(3,5))      # shortest path will be in North-East direction, could do EAST or NORTH first

print(x1)

x2=dirs_to(Point(2,2), Point(15,2))     # shortest path will be WEST due to wrap around board

print(x2)

x3=dirs_to(Point(15,15), Point(15,2))   # shortest path will be NORTH due to wrap around board

print(x3)
