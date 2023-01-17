def isOverlap(cord1,cord2):

    

    if (x1+cord1[2]<=a1 or a1+cord2[2]<=x1 or y1+cord1[3]<=b1 or b1+cord2[3]<=y1):

        return False

    

    else:

        return True
def excludeIOU(cord1,cord2):

    x1 = cord1[0]

    y1 = cord1[1]

    x2 = x1+cord1[2]

    y2 = y1+cord1[3]

    

    a1 = cord2[0]

    b1 = cord2[1]

    a2 = a1+cord2[2]

    b2 = b1+cord2[3]

    

    if(not isOverlap):

        return cord1[2]*cord1[3] +cord2[2]*cord2[3]

    

    mx1 = max(x1,a1)

    my1 = max(y1,b1)

    mx2 = min(x2,a2)

    my2 = min(y2,b2)

    

    area = (mx2 - mx1) * (my2-my1)

    return cord1[2]*cord1[3] +cord2[2]*cord2[3] - 2*area
cord1 = [0,0,4,6]

cord2 = [4,3,1,5]



excludeIOU(cord1,cord2)
cord1 = [0,0,4,6]

cord2 = [3,4,1,5]



excludeIOU(cord1,cord2)
cord1 = [6,1,5,2]

cord2 = [3,2,9,2]



excludeIOU(cord1,cord2)