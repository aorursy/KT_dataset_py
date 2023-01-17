%%writefile submission.py

global MinDistance
global targets
global globalShipState
global changeDirection
global halite2objectsRatioA
global scruteArea
global minHaliteToMove
global factorToMove
targets={}
globalShipState={}


scruteArea=6
minDistance=8
halite2objectsRatio =1800
minHaliteToMove=40 
factorToMove=2

####################
# Helper functions #
####################

# Helper function we'll use for getting adjacent position with the most halite
def argmax(arr, key=None):
    return arr.index(max(arr, key=key)) if key else arr.index(max(arr))

# Converts position from 1D to 2D representation
def get_col_row(size, pos):
    return (pos % size, pos // size)

# Returns the position in some direction relative to the current position (pos) 
def get_to_pos(size, pos, direction):
    col, row = get_col_row(size, pos)
    if direction == "NORTH":
        return pos - size if pos >= size else size ** 2 - size + col
    elif direction == "SOUTH":
        return col if pos + size >= size ** 2 else pos + size
    elif direction == "EAST":
        return pos + 1 if col < size - 1 else row * size
    elif direction == "WEST":
        return pos - 1 if col > 0 else (row + 1) * size - 1

# Get positions in all directions relative to the current position (pos)
# Especially useful for figuring out how much halite is around you
def getAdjacent(pos, size):
    return [
        get_to_pos(size, pos, "NORTH"),
        get_to_pos(size, pos, "SOUTH"),
        get_to_pos(size, pos, "EAST"),
        get_to_pos(size, pos, "WEST"),
    ]

def check_cell(j, i, game_map, player, size, debug=False):
    occupied = False
    ennemy=None
    cargo=0
    pos=i*size+j
    if debug:
        print(game_map[j%size][i%size]["ship"])
    #print ("shipyard : ", game_map[j%size][i%size]["shipyard"], " at ", pos, "(x,y)", j, i)
    #if game_map[j%size][i%size]["shipyard"]!=None and game_map[j%size][i%size]["shipyard"]!=0: print ("shipyard : ", game_map[j%size][i%size]["shipyard"], " at ", pos, "(x,y)", j, i)
    if game_map[j%size][i%size]["ship"]!= None and  game_map[j%size][i%size]["ship"]!=player:
        ennemy="ship"
        cargo=game_map[j%size][i%size]["ship_cargo"]
        occupied=True
    if game_map[j%size][i%size]["shipyard"]!= None and game_map[j%size][i%size]["shipyard"]!=player:
        ennemy="shipyard"
        #print("found ennemy shipyard at ", j, i, "means", i*size+j)
        occupied=True
    if game_map[j%size][i%size]["ship"]!= None and  game_map[j%size][i%size]["ship"]==player:
        ennemy="self"
        print("self at ", pos, j, i)
        occupied=True
    return ennemy, cargo

# Returns best direction to move from one position (fromPos) to another (toPos)
# Example: If I'm at pos 0 and want to get to pos 55, which direction should I choose?
def getDirTo(fromPos, toPos, game_map,player, size):
    fromY, fromX = divmod(fromPos, size)
    toY,   toX   =  divmod(toPos,   size)
    nennemy, ncargo=check_cell(fromX,fromY-1, game_map, player, size )
    sennemy, scargo=check_cell(fromX,fromY+1, game_map, player, size )
    eennemy, ecargo=check_cell(fromX+1,fromY, game_map, player, size )
    wennemy, wcargo=check_cell(fromX-1,fromY, game_map, player, size )
    #print(nennemy, sennemy , eennemy, wennemy)
    if (toY-fromY)%size < (fromY-toY)%size :
        if (sennemy=="ship" and scargo>game_map[fromX][fromY]["ship_cargo"]) or sennemy==None:
            return "SOUTH"
        else: return "ENNEMY"
    if (toY-fromY)%size > (fromY-toY)%size :
        if (nennemy=="ship" and ncargo>game_map[fromX][fromY]["ship_cargo"]) or nennemy==None:
            return "NORTH"
        else: return "ENNEMY"
    if (toX-fromX)%size < (fromX-toX)%size :
        if (eennemy=="ship" and ecargo>game_map[fromX][fromY]["ship_cargo"]) or eennemy==None:
            return "EAST"
        else: return "ENNEMY"
    if (toX-fromX)%size > (fromX-toX)%size :
        if (wennemy=="ship" and wcargo>game_map[fromX][fromY]["ship_cargo"]) or wennemy==None:
            return "WEST"
        else: return "ENNEMY"
        
    

# Possible directions a ship can move in
DIRS = ["NORTH", "SOUTH", "EAST", "WEST"]
# We'll use this to keep track of whether a ship is collecting halite or 
# carrying its cargo to a shipyard
ship_states = {}


#########
# distance
#########

def distance(currPos, targetPos, size):
    currX, currY=divmod(currPos, size)
    targetX, targetY=divmod(targetPos, size)
    diffX=abs(targetX-currX)%size
    diffY=abs(targetY-currY)%size
    
    return diffX+diffY



##################
#scrutinize
#################

def scrutinize(currPos, radius, game_map,player, size):
    currY, currX=divmod(currPos, size)
    #print("CurrX {1} CurrY {2} Radius {2} size {4}", currX, currY, radius, size)
    maxHalite=0
    xmax=(currX)%size
    ymax=(currY)%size
    for i in range(currY-radius, currY+radius):
        for j in range(currX-radius, currX+radius):
            #print(game_map[j][i]["halite"])
            if game_map[j%size][i%size]["halite"] > maxHalite:
                if game_map[j%size][i%size]["shipyard"] == None and game_map[j%size][i%size]["ship"]== None:
                    xmax=j%size
                    ymax=i%size
                    maxHalite=game_map[j%size][i%size]["halite"]
            if game_map[j%size][i%size]["ship"]!= None and  game_map[j%size][i%size]["ship"]!=player and game_map[j%size][i%size]["shipyard"]== None:
                cargo=game_map[j%size][i%size]["ship_cargo"]
                if cargo>maxHalite and cargo > game_map[currX%size][currY%size]["ship_cargo"]:
                    xmax=j%size
                    ymax=i%size
                    maxHalite=cargo
        
    #print (currPos,ymax*size+xmax,  xmax, ymax, maxHalite)            
    return ymax*size+xmax, maxHalite

def detect_ennemy(pos, game_map, player, status, direction, size):
    fromX, fromY =divmod(pos, size)
    newdir=None
    playerhalite=game_map[fromX][fromY]["ship_cargo"]
    if playerhalite==None: playerhalite=0
    staticAction={"ENNEMY": {"NORTH": "SOUTH", "WEST": "EAST", "EAST": "WEST", "SOUTH": "NORTH"}, "SELF": "NOTHING"}
    movingAction={"ENNEMY": {"NORTH": "SOUTH", "WEST": "EAST", "EAST": "WEST", "SOUTH": "NORTH"}, 
                  "SELF": {"NORTH": "SOUTH", "WEST": "EAST", "EAST": "WEST", "SOUTH": "NORTH"}}
    i=0
    if status != "MOVING":
        nennemy, ncargo=check_cell(fromX,fromY-1, game_map, player, size, True )
        sennemy, scargo=check_cell(fromX,fromY+1, game_map, player, size, True)
        eennemy, ecargo=check_cell(fromX+1,fromY, game_map, player, size, True )
        wennemy, wcargo=check_cell(fromX-1,fromY, game_map, player, size, True )
        
        if ncargo == None: ncargo=0
        if scargo == None: scargo=0
        if ecargo == None: ecargo=0
        if wcargo == None: wcargo=0    
            
        if nennemy =="ship" and ncargo<playerhalite:
            newdir="SOUTH"
            i+=1
        if sennemy  =="ship" and scargo<playerhalite:
            newdir="ship"
            i+=1
        if wennemy  =="ship" and wcargo<playerhalite:
            newdir="EAST"
            i+=1
        if eennemy  =="ship" and ecargo<playerhalite:
            newdir="WEST"
            i+=1
        print(sennemy, wennemy, eennemy, nennemy)
    else:
        if direction == "NORTH":
            NEennemy, cargo=check_cell(fromX-1,fromY-1, game_map, player, size )
            NWennemy, cargo=check_cell(fromX+1,fromY-1, game_map, player, size )
            Nennemy, cargo=check_cell(fromX,fromY-2, game_map, player, size )
            print("***", NEennemy, NWennemy, Nennemy)
            if "ship" in (NEennemy, NWennemy, Nennemy):# or "self" in (NEennemy, NWennemy, Nennemy):
                newdir="SOUTH"
                
        if direction == "WEST":
            SWennemy, cargo=check_cell(fromX-1,fromY+1, game_map, player, size )
            NWennemy, cargo=check_cell(fromX-1,fromY-1, game_map, player, size )
            Wennemy, cargo=check_cell(fromX-2,fromY, game_map, player, size )
            if "ship" in (Wennemy, NWennemy, SWennemy):# or "self"  in (Wennemy, NWennemy, SWennemy):
                newdir="EAST"
        if direction == "EAST":
            NEennemy, cargo=check_cell(fromX+1,fromY-1, game_map, player, size )
            SEennemy, cargo=check_cell(fromX+1,fromY+1, game_map, player, size )
            Eennemy, cargo=check_cell(fromX+2,fromY, game_map, player, size )
            if "ship" in (NEennemy, SEennemy, Eennemy):# or "self" in (NEennemy, SEennemy, Eennemy):
                newdir="WEST"
        if direction == "SOUTH":
            SEennemy, cargo=check_cell(fromX+1,fromY+1, game_map, player, size )
            SWennemy, cargo=check_cell(fromX-1,fromY+1, game_map, player, size )
            Sennemy, cargo=check_cell(fromX,fromY+2, game_map, player, size )
            if "ship" in (SEennemy, SWennemy, Sennemy):# or "self" in (SEennemy, SWennemy, Sennemy):
                newdir="NORTH"            
        print(direction, newdir, cargo)
    if i>1: 
        newdir="CONVERT"
        print("Convert")
    else:
        if newdir != None:
            print(newdir)
    
    return newdir 
    

def get_map(obs, conf):
    # define map
    game_map = []
    for x in range(conf.size):
        game_map.append([])
        for y in range(conf.size):
            game_map[x].append({
                # value will be ID of owner
                "shipyard": None,
                # value will be ID of owner
                "ship": None,
                # value will be amount of halite
                "ship_cargo": None,
                # amount of halite
                "halite": obs.halite[conf.size * y + x]})

    # place ships and shipyards on the map
    for player in range(len(obs.players)):
        shipyards = list(obs.players[player][1].values())
        for shipyard in shipyards:
            x = shipyard % conf.size
            y = shipyard // conf.size
            # place shipyard on the map            
            game_map[x][y]["shipyard"] = player
            #print(game_map[x][y]["shipyard"])

        ships = list(obs.players[player][2].values())
        for ship in ships:
            x = ship[0] % conf.size
            y = ship[0] // conf.size
            # place ship on the map
            game_map[x][y]["ship"] = player
            game_map[x][y]["ship_cargo"] = ship[1]
    return game_map


#############
# The agent #
#############

def agent(obs, config):
    global globalShipState
    global targets 
    
    
    
    # Get the player's halite, shipyard locations, and ships (along with cargo) 
    player_halite, shipyards, ships = obs.players[obs.player]
    size = config["size"]
    # Initialize a dictionary containing commands that will be sent to the game
    action = {}
    #print(list(obs.players[0][2].values())[0][0])
    
    
    mymap=get_map(obs, config)
    MaxPos={}
    MaxHalite={}
    #if len(ships)>0:
    #    scrute=round(scruteArea/len(ships))
    #    if scrute==0: scrute=1
    
    
    # If there are no ships, use first shipyard to spawn a ship.
    if len(ships) == 0 and len(shipyards) > 0:
        uid = list(shipyards.keys())[-1]
        action[uid] = "SPAWN"
    else:
        if (obs.step%3==0) and (obs.step<150) and (player_halite/(len(ships)+len(shipyards))>halite2objectsRatio) and len(shipyards) > 0:
            # if far enough from the farthest shipyard from any ship then convert into shipyard 
            farthest=0
            i=0
            for shippos in list(shipyards.values()):                
                closest=size**2
                for shipypos in list(ships.values()):
                    spos=shipypos[1]
                    if distance(spos, shippos, size) < closest:                        
                        closest=distance(spos, shippos, size)                            
                        
                if closest > farthest:
                    farthest=closest
                    indfar=i
                i+=1                    
            
            uid = list(shipyards.keys())[indfar]
            action[uid] = "SPAWN"
        
    # If there are no shipyards, convert first ship into shipyard.
    if len(shipyards) == 0 and len(ships) > 0:
        uid = list(ships.keys())[0]
        #print (list(ships.values())[0][0])
        action[uid] = "CONVERT"
        globalShipState[uid]="CONVERTING"
        ship_states[uid]="NOTHING"

    #if obs.step%10==0: print(obs.step) 
    for uid, ship in ships.items():
        
        if uid not in action: # Ignore ships that will be converted to shipyards
            pos, cargo = ship # Get the ship's position and halite in cargo
            print("pos ", pos)
            #print("uid before assignment", uid)
            if not (uid in globalShipState):
                globalShipState[uid]='COLLECTING'
                #print("changed ", uid,  "COLLECTING")
            ### check the sheep needs to move
            #print("global ship after assignment", globalShipState[uid])
            ### Part 1: Set the ship's state 
            if cargo < 500: # If cargo is too low, collect halite
                if globalShipState[uid]=='DEPOSITING':
                    #because if it is < 500 means it is not depositing any more
                    globalShipState[uid]='COLLECTING'
                    #print ("will global ship state still be depositing ?")
                if globalShipState[uid]=='COLLECTING':
                    # keep the global status collecting unless there's a significantly
                    # bigger amount of halite in the neibourhood
                    currHalite= obs.halite[pos]
                    MaxPos[uid], MaxHalite[uid]=scrutinize(list(ships.values())[0][0], scruteArea, mymap, obs.player, config.size)
                    if MaxHalite[uid]>minHaliteToMove+currHalite*factorToMove:
                        
                        globalShipState[uid]='MOVING'
                        #print("pos is ",pos, "moving to ", MaxPos[uid])
                        targets[uid]=MaxPos[uid]
                ship_states[uid] = "COLLECT"
                if globalShipState[uid]=='ESCAPING':
                    ship_states[uid] = "DEPOSIT"
                    
           
                
            if cargo >= 500: # If cargo gets very big, deposit halite
                
                # if far enough from the closest shipyard then convert into shipyard 
                closest=size**2
                i=0                
                for shipypos in list(shipyards.values()):
                    if distance(pos, shipypos, size) < closest:
                        closest=distance(pos, shipypos, size)
                        indshyp=i
                    i+=1
                
                if ((obs.step<200) and (distance(pos, list(shipyards.values())[indshyp], size) > minDistance and player_halite / (len(ships)+len(shipyards))>3*halite2objectsRatio) and len(shipyards)<=len(ships)) or (globalShipState[uid]=='CONVERTING'):
                    action[uid] = "CONVERT"
                else:
                    ship_states[uid] = "DEPOSIT"
                
            ### Part 2: Use the ship's state to select an action
            if ship_states[uid] == "COLLECT":
                # if global status is moving go to the target defined in the current turn or in previous turns
                if globalShipState[uid]=='MOVING':
                    newPos=getDirTo(list(ships.values())[0][0], targets[uid], mymap, obs.player, config.size)
                    #print (newPos)
                    if newPos!="ENNEMY" and newPos!=None:
                        newDirection=detect_ennemy(pos, mymap, obs.player, "MOVING", newPos, size)
                        if newDirection: 
                            print("change from static to", newPos, newDirection)
                            globalShipState[uid]='ESCAPING'
                            newPos=newDirection
                        
                        action[uid]=newPos
                    else:
                        if newPos=="ENNEMY":
                            #print("Danger change from to deposit", obs.step)
                            ship_states[uid] = "DEPOSIT"
                            globalShipState[uid]='COLLECTING'
                    if newPos==None:
                        globalShipState[uid]='LANDING'
                else:
                    if globalShipState[uid]=='LANDING':
                        globalShipState[uid]='COLLECTING'
                    newDirection=detect_ennemy(pos, mymap, obs.player, "COLLECTING", "NORTH", size)
                    if globalShipState=='ESCAPING':
                        currHalite= obs.halite[pos]
                        MaxPos[uid], MaxHalite[uid]=scrutinize(list(ships.values())[0][0], scruteArea, mymap, obs.player, config.size)
                        newDirection=getDirTo(pos, MaxPos[uid], mymap, obs.player, config.size)
                    if newDirection: print("tag2", newDirection)
                    if newDirection: 
                        #print("change from static to", newDirection)
                        action[uid]=newDirection
                        globalShipState[uid]='MOVING'
            
            if ship_states[uid] == "DEPOSIT":
                if globalShipState[uid] !='DEPOSITING':
                    # Move towards the closest shipyard to deposit cargo
                    closest=size**2
                    i=0
                    for shipypos in list(shipyards.values()):
                        if distance(pos, shipypos, size) < closest:
                            closest=distance(pos, shipypos, size)
                            indshyp=i-1
                        i+=1
                    targets[uid]=list(shipyards.values())[indshyp]
                    globalShipState[uid] ='DEPOSITING'
                    direction = getDirTo(pos, targets[uid], mymap, obs.player, size)
                    #print("setting to DEPOSITING to  ", targets[uid])
                else:
                    #print("Status already in DEPOSITING setting direction to  ", targets[uid])
                    direction = getDirTo(pos, targets[uid], mymap, obs.player, size)
                
                if direction!="ENNEMY" and direction!=None:
                    newDirection=detect_ennemy(pos, mymap, obs.player, "MOVING", direction, size)
                    
                    if newDirection:
                        #print ("new direction recommended", newDirection)
                        direction = newDirection
                    action[uid] = direction
                    #print("have to deposit towards ", pos, obs.step)
                else:
                    if direction=="ENNEMY":
                        #print("Change to collecting", obs.step)
                        globalShipState[uid]='CONVERTING'
                        action[uid] ="CONVERT"

    #print ("at", obs.step, "status", globalShipState)    
    return action
        
%%writefile correct2.py

global MinDistance
global targets
global globalShipState
global changeDirection
global halite2objectsRatio
global scruteArea
global minHaliteToMove
global factorToMove
targets={}
globalShipState={}

scruteArea=4
minDistance=8
halite2objectsRatio =1500
minHaliteToMove=200 
factorToMove=1.5
####################
# Helper functions #
####################

# Helper function we'll use for getting adjacent position with the most halite
def argmax(arr, key=None):
    return arr.index(max(arr, key=key)) if key else arr.index(max(arr))

# Converts position from 1D to 2D representation
def get_col_row(size, pos):
    return (pos % size, pos // size)

# Returns the position in some direction relative to the current position (pos) 
def get_to_pos(size, pos, direction):
    col, row = get_col_row(size, pos)
    if direction == "NORTH":
        return pos - size if pos >= size else size ** 2 - size + col
    elif direction == "SOUTH":
        return col if pos + size >= size ** 2 else pos + size
    elif direction == "EAST":
        return pos + 1 if col < size - 1 else row * size
    elif direction == "WEST":
        return pos - 1 if col > 0 else (row + 1) * size - 1

# Get positions in all directions relative to the current position (pos)
# Especially useful for figuring out how much halite is around you
def getAdjacent(pos, size):
    return [
        get_to_pos(size, pos, "NORTH"),
        get_to_pos(size, pos, "SOUTH"),
        get_to_pos(size, pos, "EAST"),
        get_to_pos(size, pos, "WEST"),
    ]

def check_cell(j, i, game_map, player, size):
    occupied = False
    ennemy=None
    cargo=0
    pos=i*size+j
    #print(game_map[j%size][i%size])
    #print ("shipyard : ", game_map[j%size][i%size]["shipyard"], " at ", pos, "(x,y)", j, i)
    #if game_map[j%size][i%size]["shipyard"]!=None and game_map[j%size][i%size]["shipyard"]!=0: print ("shipyard : ", game_map[j%size][i%size]["shipyard"], " at ", pos, "(x,y)", j, i)
    if game_map[j%size][i%size]["ship"]!= None and  game_map[j%size][i%size]["ship"]!=player:
        ennemy="ship"
        cargo=game_map[j%size][i%size]["ship_cargo"]
        occupied=True
    if game_map[j%size][i%size]["shipyard"]!= None and game_map[j%size][i%size]["shipyard"]!=player:
        ennemy="shipyard"
        #print("found ennemy shipyard at ", j, i, "means", i*size+j)
        occupied=True
    if game_map[j%size][i%size]["ship"]!= None and  game_map[j%size][i%size]["ship"]==player:
        ennemy="self"
        occupied=True
    return ennemy, cargo

# Returns best direction to move from one position (fromPos) to another (toPos)
# Example: If I'm at pos 0 and want to get to pos 55, which direction should I choose?
def getDirTo(fromPos, toPos, game_map,player, size):
    fromY, fromX = divmod(fromPos, size)
    toY,   toX   =  divmod(toPos,   size)
    nennemy, ncargo=check_cell(fromX,fromY-1, game_map, player, size )
    sennemy, scargo=check_cell(fromX,fromY+1, game_map, player, size )
    eennemy, ecargo=check_cell(fromX+1,fromY, game_map, player, size )
    wennemy, wcargo=check_cell(fromX-1,fromY, game_map, player, size )
    #print(nennemy, sennemy , eennemy, wennemy)
    if (toY-fromY)%size < (fromY-toY)%size :
        if (sennemy=="ship" and scargo>game_map[fromX][fromY]["ship_cargo"]) or sennemy==None:
            return "SOUTH"
        else: return "ENNEMY"
    if (toY-fromY)%size > (fromY-toY)%size :
        if (nennemy=="ship" and ncargo>game_map[fromX][fromY]["ship_cargo"]) or nennemy==None:
            return "NORTH"
        else: return "ENNEMY"
    if (toX-fromX)%size < (fromX-toX)%size :
        if (eennemy=="ship" and ecargo>game_map[fromX][fromY]["ship_cargo"]) or eennemy==None:
            return "EAST"
        else: return "ENNEMY"
    if (toX-fromX)%size > (fromX-toX)%size :
        if (wennemy=="ship" and wcargo>game_map[fromX][fromY]["ship_cargo"]) or wennemy==None:
            return "WEST"
        else: return "ENNEMY"
        
    

# Possible directions a ship can move in
DIRS = ["NORTH", "SOUTH", "EAST", "WEST"]
# We'll use this to keep track of whether a ship is collecting halite or 
# carrying its cargo to a shipyard
ship_states = {}


#########
# distance
#########

def distance(currPos, targetPos, size):
    currX, currY=divmod(currPos, size)
    targetX, targetY=divmod(targetPos, size)
    diffX=abs(targetX-currX)%size
    diffY=abs(targetY-currY)%size
    
    return diffX+diffY



##################
#scrutinize
#################

def scrutinize(currPos, radius, game_map,player, size):
    currY, currX=divmod(currPos, size)
    #print("CurrX {1} CurrY {2} Radius {2} size {4}", currX, currY, radius, size)
    maxHalite=0
    xmax=(currX)%size
    ymax=(currY)%size
    for i in range(currY-radius, currY+radius):
        for j in range(currX-radius, currX+radius):
            #print(game_map[j][i]["halite"])
            if game_map[j%size][i%size]["halite"] > maxHalite:
                if game_map[j%size][i%size]["shipyard"] == None and game_map[j%size][i%size]["ship"]== None:
                    xmax=j%size
                    ymax=i%size
                    maxHalite=game_map[j%size][i%size]["halite"]
            if game_map[j%size][i%size]["ship"]!= None and  game_map[j%size][i%size]["ship"]!=player and game_map[j%size][i%size]["shipyard"]== None:
                cargo=game_map[j%size][i%size]["ship_cargo"]
                if cargo>maxHalite and cargo > game_map[currX%size][currY%size]["ship_cargo"]:
                    xmax=j%size
                    ymax=i%size
                    maxHalite=cargo
        
    #print (currPos,ymax*size+xmax,  xmax, ymax, maxHalite)            
    return ymax*size+xmax, maxHalite

def get_map(obs, conf):
    # define map
    game_map = []
    for x in range(conf.size):
        game_map.append([])
        for y in range(conf.size):
            game_map[x].append({
                # value will be ID of owner
                "shipyard": None,
                # value will be ID of owner
                "ship": None,
                # value will be amount of halite
                "ship_cargo": None,
                # amount of halite
                "halite": obs.halite[conf.size * y + x]})

    # place ships and shipyards on the map
    for player in range(len(obs.players)):
        shipyards = list(obs.players[player][1].values())
        for shipyard in shipyards:
            x = shipyard % conf.size
            y = shipyard // conf.size
            # place shipyard on the map            
            game_map[x][y]["shipyard"] = player
            #print(game_map[x][y]["shipyard"])

        ships = list(obs.players[player][2].values())
        for ship in ships:
            x = ship[0] % conf.size
            y = ship[0] // conf.size
            # place ship on the map
            game_map[x][y]["ship"] = player
            game_map[x][y]["ship_cargo"] = ship[1]
    return game_map


#############
# The agent #
#############

def agent(obs, config):
    global globalShipState
    global targets 
    
    
    
    # Get the player's halite, shipyard locations, and ships (along with cargo) 
    player_halite, shipyards, ships = obs.players[obs.player]
    size = config["size"]
    # Initialize a dictionary containing commands that will be sent to the game
    action = {}
    #print(list(obs.players[0][2].values())[0][0])
    
    
    mymap=get_map(obs, config)
    MaxPos={}
    MaxHalite={}
    #if len(ships)>0:
    #    scrute=round(scruteArea/len(ships))
    #    if scrute==0: scrute=1
    
    
    # If there are no ships, use first shipyard to spawn a ship.
    if len(ships) == 0 and len(shipyards) > 0:
        uid = list(shipyards.keys())[-1]
        action[uid] = "SPAWN"
    else:
        if (obs.step%30==0) and (obs.step<250) and (player_halite/(len(ships)+len(shipyards))>halite2objectsRatio) and len(shipyards) > 0:
            # if far enough from the farthest shipyard from any ship then convert into shipyard 
            farthest=0
            i=0
            for shippos in list(shipyards.values()):                
                closest=size**2
                for shipypos in list(ships.values()):
                    spos=shipypos[1]
                    if distance(spos, shippos, size) < closest:                        
                        closest=distance(spos, shippos, size)                            
                        
                if closest > farthest:
                    farthest=closest
                    indfar=i
                i+=1                    
            
            uid = list(shipyards.keys())[indfar]
            action[uid] = "SPAWN"
        
    # If there are no shipyards, convert first ship into shipyard.
    if len(shipyards) == 0 and len(ships) > 0:
        uid = list(ships.keys())[0]
        #print (list(ships.values())[0][0])
        action[uid] = "CONVERT"
        globalShipState[uid]="CONVERTING"
        ship_states[uid]="NOTHING"

    #if obs.step%10==0: print(obs.step) 
    for uid, ship in ships.items():
        
        if uid not in action: # Ignore ships that will be converted to shipyards
            pos, cargo = ship # Get the ship's position and halite in cargo
            #print("uid before assignment", uid)
            if not (uid in globalShipState):
                globalShipState[uid]='NEW'
                #print("changed ", uid,  "COLLECTING")
            ### check the sheep needs to move
            #print("global ship after assignment", globalShipState[uid])
            ### Part 1: Set the ship's state 
            if cargo < 500: # If cargo is too low, collect halite
                if globalShipState[uid]=='DEPOSITING':
                    #because if it is < 500 means it is not depositing any more
                    globalShipState[uid]='COLLECTING'
                    #print ("will global ship state still be depositing ?")
                if globalShipState[uid] in ['COLLECTING', 'NEW']:
                    # keep the global status collecting unless there's a significantly
                    # bigger amount of halite in the neibourhood
                    if globalShipState[uid]:
                        currHalite= obs.halite[pos]
                    else:
                        currHalite=0
                    MaxPos[uid], MaxHalite[uid]=scrutinize(list(ships.values())[0][0], scruteArea, mymap, obs.player, config.size)
                    if MaxHalite[uid]>minHaliteToMove+currHalite*factorToMove:
                        
                        globalShipState[uid]='MOVING'
                        #print("pos is ",pos, "moving to ", MaxPos[uid])
                        targets[uid]=MaxPos[uid]
                ship_states[uid] = "COLLECT"
                    
           
                
            if cargo >= 500: # If cargo gets very big, deposit halite
                
                # if far enough from the closest shipyard then convert into shipyard 
                closest=size**2
                i=0                
                for shipypos in list(shipyards.values()):
                    if distance(pos, shipypos, size) < closest:
                        closest=distance(pos, shipypos, size)
                        indshyp=i
                    i+=1
                
                if ((obs.step<200) and (distance(pos, list(shipyards.values())[indshyp], size) > minDistance and player_halite / (len(ships)+len(shipyards))>2*halite2objectsRatio) and len(shipyards)<=len(ships)) or (globalShipState[uid]=='CONVERTING'):
                    action[uid] = "CONVERT"
                else:
                    ship_states[uid] = "DEPOSIT"
                
            ### Part 2: Use the ship's state to select an action
            if ship_states[uid] == "COLLECT":
                # if global status is moving go to the target defined in the current turn or in previous turns
                if globalShipState[uid]=='MOVING':
                    newPos=getDirTo(list(ships.values())[0][0], targets[uid], mymap, obs.player, config.size)
                    #print (newPos)
                    if newPos!="ENNEMY" and newPos!=None:
                        action[uid]=newPos
                    else:
                        if newPos=="ENNEMY":
                            #print("Danger change from to deposit", obs.step)
                            ship_states[uid] = "DEPOSIT"
                            globalShipState[uid]='COLLECTING'
                    if newPos==None:
                        globalShipState[uid]='LANDING'
                else:
                    if globalShipState[uid]=='LANDING':
                        globalShipState[uid]='COLLECTING'

            
            if ship_states[uid] == "DEPOSIT":
                if globalShipState[uid] !='DEPOSITING':
                    # Move towards the closest shipyard to deposit cargo
                    closest=size**2
                    i=0
                    for shipypos in list(shipyards.values()):
                        if distance(pos, shipypos, size) < closest:
                            closest=distance(pos, shipypos, size)
                            indshyp=i-1
                        i+=1
                    targets[uid]=list(shipyards.values())[indshyp]
                    globalShipState[uid] ='DEPOSITING'
                    direction = getDirTo(pos, targets[uid], mymap, obs.player, size)
                    #print("setting to DEPOSITING to  ", targets[uid])
                else:
                    #print("Status already in DEPOSITING setting direction to  ", targets[uid])
                    direction = getDirTo(pos, targets[uid], mymap, obs.player, size)
                
                if direction!="ENNEMY" and direction!=None:                     
                    action[uid] = direction
                    #print("have to deposit towards ", pos, obs.step)
                else:
                    if direction=="ENNEMY":
                        
                        #print("Change to collecting", obs.step)
                        globalShipState[uid]='CONVERTING'

    #print ("at", obs.step, "status", globalShipState)    
    return action
        
%%writefile "best.py"

global MinDistance
global targets
global globalShipState
global changeDirection
global halite2objectsRatio
global scruteArea
global minHaliteToMove
global factorToMove
targets={}
globalShipState={}

scruteArea=3
minDistance=8
halite2objectsRatio =1800
minHaliteToMove=150 
factorToMove=1.5
####################
# Helper functions #
####################

# Helper function we'll use for getting adjacent position with the most halite
def argmax(arr, key=None):
    return arr.index(max(arr, key=key)) if key else arr.index(max(arr))

# Converts position from 1D to 2D representation
def get_col_row(size, pos):
    return (pos % size, pos // size)

# Returns the position in some direction relative to the current position (pos) 
def get_to_pos(size, pos, direction):
    col, row = get_col_row(size, pos)
    if direction == "NORTH":
        return pos - size if pos >= size else size ** 2 - size + col
    elif direction == "SOUTH":
        return col if pos + size >= size ** 2 else pos + size
    elif direction == "EAST":
        return pos + 1 if col < size - 1 else row * size
    elif direction == "WEST":
        return pos - 1 if col > 0 else (row + 1) * size - 1

# Get positions in all directions relative to the current position (pos)
# Especially useful for figuring out how much halite is around you
def getAdjacent(pos, size):
    return [
        get_to_pos(size, pos, "NORTH"),
        get_to_pos(size, pos, "SOUTH"),
        get_to_pos(size, pos, "EAST"),
        get_to_pos(size, pos, "WEST"),
    ]

def check_cell(j, i, game_map, player, size):
    occupied = False
    ennemy=None
    cargo=0
    pos=i*size+j
    #print(game_map[j%size][i%size])
    #print ("shipyard : ", game_map[j%size][i%size]["shipyard"], " at ", pos, "(x,y)", j, i)
    #if game_map[j%size][i%size]["shipyard"]!=None and game_map[j%size][i%size]["shipyard"]!=0: print ("shipyard : ", game_map[j%size][i%size]["shipyard"], " at ", pos, "(x,y)", j, i)
    if game_map[j%size][i%size]["ship"]!= None and  game_map[j%size][i%size]["ship"]!=player:
        ennemy="ship"
        cargo=game_map[j%size][i%size]["ship_cargo"]
        occupied=True
    if game_map[j%size][i%size]["shipyard"]!= None and game_map[j%size][i%size]["shipyard"]!=player:
        ennemy="shipyard"
        print("found ennemy shipyard at ", j, i, "means", i*size+j)
        occupied=True
    if game_map[j%size][i%size]["ship"]!= None and  game_map[j%size][i%size]["ship"]==player:
        ennemy="self"
        occupied=True
    return ennemy, cargo

# Returns best direction to move from one position (fromPos) to another (toPos)
# Example: If I'm at pos 0 and want to get to pos 55, which direction should I choose?
def getDirTo(fromPos, toPos, game_map,player, size):
    fromY, fromX = divmod(fromPos, size)
    toY,   toX   =  divmod(toPos,   size)
    nennemy, ncargo=check_cell(fromX,fromY-1, game_map, player, size )
    sennemy, scargo=check_cell(fromX,fromY+1, game_map, player, size )
    eennemy, ecargo=check_cell(fromX+1,fromY, game_map, player, size )
    wennemy, wcargo=check_cell(fromX-1,fromY, game_map, player, size )
    #print(nennemy, sennemy , eennemy, wennemy)
    if (toY-fromY)%size < (fromY-toY)%size :
        if (sennemy=="ship" and scargo>game_map[fromX][fromY]["ship_cargo"]) or sennemy==None:
            return "SOUTH"
        else: return "ENNEMY"
    if (toY-fromY)%size > (fromY-toY)%size :
        if (nennemy=="ship" and ncargo>game_map[fromX][fromY]["ship_cargo"]) or nennemy==None:
            return "NORTH"
        else: return "ENNEMY"
    if (toX-fromX)%size < (fromX-toX)%size :
        if (eennemy=="ship" and ecargo>game_map[fromX][fromY]["ship_cargo"]) or eennemy==None:
            return "EAST"
        else: return "ENNEMY"
    if (toX-fromX)%size > (fromX-toX)%size :
        if (wennemy=="ship" and wcargo>game_map[fromX][fromY]["ship_cargo"]) or wennemy==None:
            return "WEST"
        else: return "ENNEMY"
        
    

# Possible directions a ship can move in
DIRS = ["NORTH", "SOUTH", "EAST", "WEST"]
# We'll use this to keep track of whether a ship is collecting halite or 
# carrying its cargo to a shipyard
ship_states = {}


#########
# distance
#########

def distance(currPos, targetPos, size):
    currX, currY=divmod(currPos, size)
    targetX, targetY=divmod(targetPos, size)
    diffX=abs(targetX-currX)%size
    diffY=abs(targetY-currY)%size
    
    return diffX+diffY



##################
#scrutinize
#################

def scrutinize(currPos, radius, game_map,player, size):
    currY, currX=divmod(currPos, size)
    #print("CurrX {1} CurrY {2} Radius {2} size {4}", currX, currY, radius, size)
    maxHalite=0
    xmax=(currX)%size
    ymax=(currY)%size
    for i in range(currY-radius, currY+radius):
        for j in range(currX-radius, currX+radius):
            #print(game_map[j][i]["halite"])
            if game_map[j%size][i%size]["halite"] > maxHalite:
                if game_map[j%size][i%size]["shipyard"] == None and game_map[j%size][i%size]["ship"]== None:
                    xmax=j%size
                    ymax=i%size
                    maxHalite=game_map[j%size][i%size]["halite"]
            if game_map[j%size][i%size]["ship"]!= None and  game_map[j%size][i%size]["ship"]!=player and game_map[j%size][i%size]["shipyard"]== None:
                cargo=game_map[j%size][i%size]["ship_cargo"]
                if cargo>maxHalite and cargo > game_map[currX%size][currY%size]["ship_cargo"]:
                    xmax=j%size
                    ymax=i%size
                    maxHalite=cargo
        
    #print (currPos,ymax*size+xmax,  xmax, ymax, maxHalite)            
    return ymax*size+xmax, maxHalite

def get_map(obs, conf):
    # define map
    game_map = []
    for x in range(conf.size):
        game_map.append([])
        for y in range(conf.size):
            game_map[x].append({
                # value will be ID of owner
                "shipyard": None,
                # value will be ID of owner
                "ship": None,
                # value will be amount of halite
                "ship_cargo": None,
                # amount of halite
                "halite": obs.halite[conf.size * y + x]})

    # place ships and shipyards on the map
    for player in range(len(obs.players)):
        shipyards = list(obs.players[player][1].values())
        for shipyard in shipyards:
            x = shipyard % conf.size
            y = shipyard // conf.size
            # place shipyard on the map            
            game_map[x][y]["shipyard"] = player
            print(game_map[x][y]["shipyard"])

        ships = list(obs.players[player][2].values())
        for ship in ships:
            x = ship[0] % conf.size
            y = ship[0] // conf.size
            # place ship on the map
            game_map[x][y]["ship"] = player
            game_map[x][y]["ship_cargo"] = ship[1]
    return game_map


#############
# The agent #
#############

def agent(obs, config):
    global globalShipState
    global targets 
    
    
    
    # Get the player's halite, shipyard locations, and ships (along with cargo) 
    player_halite, shipyards, ships = obs.players[obs.player]
    size = config["size"]
    # Initialize a dictionary containing commands that will be sent to the game
    action = {}
    #print(list(obs.players[0][2].values())[0][0])
    
    
    mymap=get_map(obs, config)
    MaxPos={}
    MaxHalite={}
    #if len(ships)>0:
    #    scrute=round(scruteArea/len(ships))
    #    if scrute==0: scrute=1
    
    
    # If there are no ships, use first shipyard to spawn a ship.
    if len(ships) == 0 and len(shipyards) > 0:
        uid = list(shipyards.keys())[-1]
        action[uid] = "SPAWN"
    else:
        if (obs.step%30==0) and (obs.step<200) and (player_halite/(len(ships)+len(shipyards))>halite2objectsRatio) and len(shipyards) > 0:
            # if far enough from the farthest shipyard from any ship then convert into shipyard 
            farthest=0
            i=0
            for shippos in list(shipyards.values()):                
                closest=size**2
                for shipypos in list(ships.values()):
                    spos=shipypos[1]
                    if distance(spos, shippos, size) < closest:                        
                        closest=distance(spos, shippos, size)                            
                        
                if closest > farthest:
                    farthest=closest
                    indfar=i
                i+=1                    
            
            uid = list(shipyards.keys())[indfar]
            action[uid] = "SPAWN"
        
    # If there are no shipyards, convert first ship into shipyard.
    if len(shipyards) == 0 and len(ships) > 0:
        uid = list(ships.keys())[0]
        #print (list(ships.values())[0][0])
        action[uid] = "CONVERT"
        globalShipState[uid]="CONVERTING"
        ship_states[uid]="NOTHING"

    #if obs.step%10==0: print(obs.step) 
    for uid, ship in ships.items():
        
        if uid not in action: # Ignore ships that will be converted to shipyards
            pos, cargo = ship # Get the ship's position and halite in cargo
            #print("uid before assignment", uid)
            if not (uid in globalShipState):
                globalShipState[uid]='COLLECTING'
                #print("changed ", uid,  "COLLECTING")
            ### check the sheep needs to move
            #print("global ship after assignment", globalShipState[uid])
            ### Part 1: Set the ship's state 
            if cargo < 500: # If cargo is too low, collect halite
                if globalShipState[uid]=='DEPOSITING':
                    #because if it is < 500 means it is not depositing any more
                    globalShipState[uid]='COLLECTING'
                    print ("will global ship state still be depositing ?")
                if globalShipState[uid]=='COLLECTING':
                    # keep the global status collecting unless there's a significantly
                    # bigger amount of halite in the neibourhood
                    currHalite= obs.halite[pos]
                    MaxPos[uid], MaxHalite[uid]=scrutinize(list(ships.values())[0][0], scruteArea, mymap, obs.player, config.size)
                    if MaxHalite[uid]>minHaliteToMove+currHalite*factorToMove:
                        
                        globalShipState[uid]='MOVING'
                        #print("pos is ",pos, "moving to ", MaxPos[uid])
                        targets[uid]=MaxPos[uid]
                ship_states[uid] = "COLLECT"
                    
           
                
            if cargo >= 500: # If cargo gets very big, deposit halite
                
                # if far enough from the closest shipyard then convert into shipyard 
                closest=size**2
                i=0                
                for shipypos in list(shipyards.values()):
                    if distance(pos, shipypos, size) < closest:
                        closest=distance(pos, shipypos, size)
                        indshyp=i
                    i+=1
                
                if ((obs.step<200) and (distance(pos, list(shipyards.values())[indshyp], size) > minDistance and player_halite / (len(ships)+len(shipyards))>2*halite2objectsRatio) and len(shipyards)<=len(ships)) or (globalShipState[uid]=='CONVERTING'):
                    action[uid] = "CONVERT"
                else:
                    ship_states[uid] = "DEPOSIT"
                
            ### Part 2: Use the ship's state to select an action
            if ship_states[uid] == "COLLECT":
                # if global status is moving go to the target defined in the current turn or in previous turns
                if globalShipState[uid]=='MOVING':
                    newPos=getDirTo(list(ships.values())[0][0], targets[uid], mymap, obs.player, config.size)
                    #print (newPos)
                    if newPos!="ENNEMY" and newPos!=None:
                        action[uid]=newPos
                    else:
                        if newPos=="ENNEMY":
                            print("Danger change from to deposit", obs.step)
                            ship_states[uid] = "DEPOSIT"
                            globalShipState[uid]='COLLECTING'
                    if newPos==None:
                        globalShipState[uid]='LANDING'
                else:
                    if globalShipState[uid]=='LANDING':
                        globalShipState[uid]='COLLECTING'

            
            if ship_states[uid] == "DEPOSIT":
                if globalShipState[uid] !='DEPOSITING':
                    # Move towards the closest shipyard to deposit cargo
                    closest=size**2
                    i=0
                    for shipypos in list(shipyards.values()):
                        if distance(pos, shipypos, size) < closest:
                            closest=distance(pos, shipypos, size)
                            indshyp=i-1
                        i+=1
                    targets[uid]=list(shipyards.values())[indshyp]
                    globalShipState[uid] ='DEPOSITING'
                    direction = getDirTo(pos, targets[uid], mymap, obs.player, size)
                    #print("setting to DEPOSITING to  ", targets[uid])
                else:
                    #print("Status already in DEPOSITING setting direction to  ", targets[uid])
                    direction = getDirTo(pos, targets[uid], mymap, obs.player, size)
                
                if direction!="ENNEMY" and direction!=None:                     
                    action[uid] = direction
                    #print("have to deposit towards ", pos, obs.step)
                else:
                    if direction=="ENNEMY":
                        
                        print("Change to collecting", obs.step)
                        globalShipState[uid]='CONVERTING'

    #print ("at", obs.step, "status", globalShipState)    
    return action
        
from kaggle_environments import make, evaluate
env = make("halite", debug=True)

env.run(["submission.py", "correct2.py", "best.py", "best.py"])
env.render(mode="ipython", width=480, height=360)
