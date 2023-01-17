genres = ("pop", "rock", "soul", "hard rock", "soft rock", \
                "R&B", "progressive rock", "disco") 
print(f'Genres: {genres}')
print(f'Type of genres: {type(genres)}')
print(f'Length of genres: {len(genres)}')
genres[0]
genres[-1]
disco = ("disco",10,1.2)
rock = ("rock",10)

disco_rock = disco + rock

print(f'Disco: {disco}')
print(f'Rock: {rock}')
print(f'Disco Rock: {disco_rock}')
genres[3:6]
genres[0:2]
genres.index('disco')
ratings = (0, 9, 6, 5, 10, 8, 9, 6, 2)
ratings
ratings_sorted = tuple(sorted(ratings))
ratings_sorted
ratings_reversed = tuple(sorted(ratings,reverse=True))
ratings_reversed
# ratings[2] = 4
# TypeError: 'tuple' object does not support item assignment
nested = (1,2,("pop","rock"),(3,4),("disco",(1,2)))
nested
nested[2]
nested[2][1]
nested[2][1][0]
a_list = ["John Smith",10.1,1982]
print(f'a_list: {a_list}')
print(f'Type of a_list: {type(a_list)}')
print(f'Length of a_list: {len(a_list)}')
a_list[1]
a_list = ["John Smith",10.1,1982]
b_list = ["Jane Smith",9.1,1988]

ab_list = a_list + b_list

print(f'a_list: {a_list}')
print(f'b_list: {b_list}')
print(f'ab_list : {ab_list}')
a_list_added = a_list + ["pop",10]
a_list_added
a_list = ["John Smith",10.1,1982]
a_list
a_list.extend(["pop",10])
a_list
a_list = ["John Smith",10.1,1982]
a_list
a_list.append(["pop",10])
a_list
A = ["disco",10,1.2]
A
A[0] = "hard rock"
A
del(A[0])
A
"hard rock".split()
"A,B,C,D".split(",")
A = ["hard rock",10,1.2]
A
B = A
B
A[0]
B[0]
A[0] = "banana"
A[0]
B[0]
A = ["hard rock",10,1.2]
A
B = A[:]
B
A[0]
B[0]
A[0] = "banana"
A[0]
B[0]
album_list = ['Michael Jackson',"Thriller","Thriller",1982]

print(f'Album list: {album_list}')
print(f'Type of Album list: {type(album_list)}')
print(f'Length of Album list: {len(album_list)}')
album_set = set(album_list)
     
print(f'Album set: {album_set}')
print(f'Type of Album set: {type(album_set)}')
print(f'Length of Album set: {len(album_set)}')
A = {"Thriller","Back in Black","AC/DC"}
A
A.add("NSYNC")
A
A.remove("NSYNC")
A
"AC/DC" in A
"Who" in A
album_set_1 = {'AC/DC', 'Back in Black', 'Thriller'}
album_set_1
album_set_2= {'AC/DC', 'Back in Black', 'The Dark Side of the Moon'}
album_set_2
intersection = album_set_1 & album_set_2
intersection
album_set_4 = album_set_1.union(album_set_2)
album_set_4
intersection.issubset(album_set_1)
album_set_4.issubset(album_set_1)
album_set_4.issuperset(album_set_1)
album_set_1.difference(album_set_2)  
album_set_2.difference(album_set_1)  
album_set_1.intersection(album_set_2)  
release_year = {"Thriller": "1982", 
                "Back in Black": "1980", 
                "The Dark Side of the Moon": "1973", 
                "The Bodyguard": "1992", 
                "Bat Out of Hell": "1977", 
                "Their Greatest Hits (1971-1975)": "1976", 
                "Saturday Night Fever": "1977", 
                "Rumours": "1977"}

print(f'Release year: {release_year}')
print(f'Type of ARelease year: {type(release_year)}')
print(f'Length of Release year: {len(release_year)}')
release_year['Thriller']
print(f'Release year keys: {release_year.keys()}')
print(f'Release year values: {release_year.values()}')
release_year['Graduation'] = '2007'
release_year
# Delete entries by key
del(release_year['Thriller'])
del(release_year['Graduation'])
release_year
'The Bodyguard' in release_year
'Starboy' in release_year