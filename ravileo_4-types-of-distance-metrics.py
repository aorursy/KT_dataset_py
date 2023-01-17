# Import required libarary
from scipy.spatial import distance

# Defining the points
point_1 = (1,2,3)
point_2 = (4,5,6)

point_1,point_2
# computing the euclidean distance
euclidean_distance = distance.euclidean(point_1,point_2)
print('Euclidean Distance b/w', point_1, 'and', point_2, 'is: ', euclidean_distance)
# computing the manhattan distance
manhattan_distance = distance.cityblock(point_1,point_2)
print('Manhattan Distance b/w', point_1, 'and', point_2, 'is: ', manhattan_distance)
# computing the minkowski distance
minkowski_distance = distance.minkowski(point_1,point_2, p=3)
print('minkowski Distance b/w', point_1, 'and', point_2, 'is: ', minkowski_distance)
# computing the minkowski distance
minkowski_distance_order_1 = distance.minkowski(point_1,point_2, p=1)
print('Minkowski Distance of order 1:',minkowski_distance_order_1, '\nManhattan Distance: ',manhattan_distance,euclidean_distance)
# computing the minkowski distance
minkowski_distance_order_1 = distance.minkowski(point_1,point_2, p=2)
print('Minkowski Distance of order 1:',minkowski_distance_order_1, '\nManhattan Distance: ',manhattan_distance)
# defining two strings
string_1 = 'euclidean'
string_2 = 'manhattan'
# computing the hamming distance
hamming_distance = distance.hamming(list(string_1), list(string_2))*len(string_1)
print('Hamming Distance b/w', string_1, 'and', string_2, 'is: ', hamming_distance)
