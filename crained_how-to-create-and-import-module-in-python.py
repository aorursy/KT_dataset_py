# We can now use our script!

# We import the script that is essentially salad.py

import salad



# in order to use salad.py you must use module name first

# which in this case is salad and then use the function we made in

#salad which was make_salad

salad.make_salad('small', 'lettuce', 'tomatoe', 'onion')

salad.make_salad('large', 'lettuce', 'tomatoe', 'onion','cheese','dressing')
from salad import make_salad as ms



ms('small', 'lettuce', 'tomatoe', 'onion')

ms('large', 'lettuce', 'tomatoe', 'onion','cheese','dressing')