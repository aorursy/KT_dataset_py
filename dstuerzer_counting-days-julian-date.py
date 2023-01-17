import numpy as np # linear algebra



def date_to_jd(y, mo, d, h, mn, s):

# y...year, mo...month, d...day, h...hour (UTC), mn...minutes, s...seconds

    date = [y, mo, d, h, mn, s]

    if date[:3] >= [1582,10,15]:

        #gregorian date

        return 367*y - (7*(y+int((mo+9)/12)))//4 - (3*(int((y+(mo-9)/7)/100)+1))//4+(275*mo)//9+d+1721028.5+h/24+mn/(24*60)+s/86400

    elif date[:3] <= [1582,10,4]:

        #julian date

        return 367*y - (7*(y+5001+int((mo-9)/7)))//4+(275*mo)//9+d+1729776.5+h/24+mn/(24*60)+s/86400

    else:

        print('Error: This date does not exist!')

        return -1



date_to_jd(1845,1,26,12,0,0)