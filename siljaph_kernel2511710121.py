import matplotlib.pyplot as plt



size = 30

range_from_size = list(range(0, size))

trend = range_from_size

weekend = [0,0,0,1,1,

           0,0,0,0,0,1,1,

           0,0,0,0,0,1,1,

           0,0,0,0,0,1,1,

           0,0,0,0]

trend_plus_weekend = [sum(x) for x in zip(trend, weekend)]



plt.plot(

    range_from_size, trend, 'y-',

    range_from_size, weekend, 'b-'

)
import matplotlib.pyplot as plt



size = 30

range_from_size = list(range(0, size))

trend = range_from_size

weekend = [0,0,0,1,1,

           0,0,0,0,0,1,1,

           0,0,0,0,0,1,1,

           0,0,0,0,0,1,1,

           0,0,0,0]

trend_plus_weekend = [sum(x) for x in zip(trend, weekend)]



plt.plot(

    range_from_size, trend, 'y-',

    range_from_size, weekend, 'b-',

    range_from_size, trend_plus_weekend, 'g-'

)
import matplotlib.pyplot as plt



size = 30

range_from_size = list(range(0, size))

trend = range_from_size

weekend = [0,0,0,1,1,

           0,0,0,0,0,1,1,

           0,0,0,0,0,1,1,

           0,0,0,0,0,1,1,

           0,0,0,0]

half_trend_plus_quadruple_weekend = [0.5 * a +  4 * b for a, b in zip(trend, weekend)]



plt.plot(

    range_from_size, trend, 'y-',

    range_from_size, weekend, 'b-',

    range_from_size, half_trend_plus_quadruple_weekend, 'g-'

)