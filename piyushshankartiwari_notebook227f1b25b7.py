#Finding Area of Intersection of two Rectangles

def isIntersect(topLeft1, bottomRight1, topLeft2, bottomRight2):
    if   (  (  topLeft1[0]  >  bottomRight2[0]  ) or  (  bottomRight1[0]  <  topLeft2[0]  )  or (  topLeft1[1] > bottomRight2[1] ) or (   bottomRight1[1]  <  topLeft2[1]   ) ):
        return False
    return True


topLeft1 = tuple(map(int, input("Enter top-left coordinates of first rectangle").split()))
l1, w1 = map(int, input("Enter the length and width of first rectangle").split())
topLeft2 = tuple(map(int, input("Enter top-left coordinates of second rectangle").split()))
l2, w2 = map(int, input("Enter the length and width of second rectangle").split())

bottomRight1 = (topLeft1[0] + l1, topLeft1[1] + w1)
bottomRight2 = (topLeft2[0] + l2, topLeft2[1] + w2)

if(isIntersect(topLeft1, bottomRight1, topLeft2, bottomRight2)):
    aoi = (min(bottomRight1[0], bottomRight2[0]) - max(topLeft1[0], topLeft2[0])) * (min(bottomRight1[1], bottomRight2[1]) - max(topLeft1[1], topLeft2[1]))
    print(aoi)
else:
    print(0)
