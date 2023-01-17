
gridSize = 7
Players = 2

cols_count = gridSize
rows_count = gridSize

Matrix = [[{'Value':0,'Color':'','Valency':0} for x in range(cols_count)] for x in range(rows_count)]

All_Colors = ['Red','Green','Blue','Yellow','Pink']
Color_List = All_Colors[0:Players]

#setting cell valency
for i in range(rows_count):
    for j in range(cols_count):
        #print(i,j)
        valency = 4 #set default
        if(i==0 or i==rows_count - 1):
            valency -= 1
            
        if(j==0 or j==cols_count - 1):
            valency -= 1
            
        Matrix[i][j]['Valency'] = valency
        
color_counter = 0
#census = {}
def boop(i,j,*iterflag):
    global color_counter
    #print(len(iterflag))
    #print(color_counter)
    if(i >= 0 and j >= 0 and ((Matrix[i][j]['Color'] == Color_List[color_counter]) or (Matrix[i][j]['Color'] == '') or (len(iterflag) == 1))): 
        Matrix[i][j]['Value'] += 1
        Matrix[i][j]['Color'] = Color_List[color_counter]
        if(Matrix[i][j]['Value'] == Matrix[i][j]['Valency']):
            Matrix[i][j]['Value'] = 0
            Matrix[i][j]['Color'] = ''
            boop(i, j-1, True) #left
            boop(i-1, j, True) #top
            boop(i, j+1, True) #right
            boop(i+1, j, True) #bottom
        if(len(iterflag) == 0):
            color_counter += 1
            if(color_counter == len(Color_List)):
                color_counter = 0
        Census()

playing_chance = []        
def Census():
    census = {}
    
    for i in range(rows_count):
        for j in range(cols_count):
            if(Matrix[i][j]['Color'] in census):
                census[Matrix[i][j]['Color']] = census[Matrix[i][j]['Color']] + 1
            else :
                census[Matrix[i][j]['Color']] = 1
                playing_chance.append(Matrix[i][j]['Color'])
                print()
    #print([x for x in Color_List and playing_chance if x not in census])
    print(census)
            
boop(0,1)

boop(0,0)

boop(1,0)

boop(0,0)
