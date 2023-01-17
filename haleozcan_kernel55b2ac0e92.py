import pandas as pd

from geopy.distance import geodesic

times_square_hotels = pd.read_csv("../input/ny-times-square-hotels/times-square-hotels.csv")

new_table= times_square_hotels.drop(times_square_hotels.index[5:41])



paramount_coord = 40.759132, -73.986348,

carter_coord= 40.757309, -73.987901,

millennium_coord= 40.756667, -73.984396,

bestwesternpresident_coord= 40.760454, -73.985655,

novotel_coord= 40.762897, -73.983683 



new_table = new_table.assign(DistancetoParamount = [0, geodesic(paramount_coord, carter_coord).kilometers,geodesic(paramount_coord, millennium_coord).kilometers,geodesic(paramount_coord, bestwesternpresident_coord).kilometers ,geodesic(paramount_coord, novotel_coord).kilometers])

new_table = new_table.assign(DistancetoCarter = [geodesic(carter_coord, paramount_coord).kilometers,0, geodesic(carter_coord, millennium_coord).kilometers,geodesic(carter_coord, bestwesternpresident_coord).kilometers ,geodesic(carter_coord, novotel_coord).kilometers])

new_table = new_table.assign(DistancetoMillennium = [geodesic(millennium_coord, paramount_coord).kilometers, geodesic(millennium_coord, carter_coord).kilometers,0, geodesic(millennium_coord, bestwesternpresident_coord).kilometers ,geodesic(millennium_coord, novotel_coord).kilometers])

new_table = new_table.assign(DistancetoBestWesternPresident = [geodesic(bestwesternpresident_coord,paramount_coord).kilometers,geodesic(bestwesternpresident_coord, carter_coord).kilometers,geodesic(bestwesternpresident_coord, millennium_coord).kilometers,0, geodesic(bestwesternpresident_coord, novotel_coord).kilometers])

new_table = new_table.assign(DistancetoNovotel = [geodesic(novotel_coord,paramount_coord).kilometers,geodesic(novotel_coord, carter_coord).kilometers,geodesic(novotel_coord, millennium_coord).kilometers,geodesic(novotel_coord, bestwesternpresident_coord).kilometers, 0])



nodes = ('Paramount', 'Carter', 'Millennium', 'Best Western President', 'Novotel')

graph = {

'Paramount':{'Carter':0.241206,'Millennium': 0.319532, 'Best Western President':0.158040, 'Novotel': 0.474811},

'Carter':{'Paramount':0.241206 ,'Millennium': 0.304440, 'Best Western President': 0.397422, 'Novotel': 0.715492},

'Millennium':{'Paramount':0.319532 ,'Carter': 0.304440, 'Best Western President': 0.433773, 'Novotel': 0.694452},

'Best Western President':{'Paramount':0.158040 ,'Carter': 0.397422, 'Millennium': 0.433773, 'Novotel': 0.318317},

'Novotel':{'Paramount':0.474811 ,'Carter': 0.715492, 'Millennium': 0.694452, 'Best Western President': 0.318317},

}



unvisited = {node: None for node in nodes} 

visited = {}

current = 'Paramount'

currentDistance = 0

unvisited[current] = currentDistance

chosen=[current]

while True:

    unvisited = {node: None for node in graph[current]}

    for neighbour, distance in graph[current].items():

        

        if neighbour in chosen: continue

       

        if neighbour not in unvisited: continue

        newDistance = currentDistance + distance

        if unvisited[neighbour] is None or unvisited[neighbour] > newDistance:

            unvisited[neighbour] = newDistance

           

    visited[current] = currentDistance

 

    if not unvisited: break

    candidates = [node for node in unvisited.items() if node[1]]

    if len(candidates) == 0:

        break

   

    current, currentDistance = sorted(candidates, key = lambda x: x[1])[0]

   

    chosen.append(current)

    

print(visited)