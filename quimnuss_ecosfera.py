# -*- coding: utf-8 -*-

# libraries

import os



from bokeh.io import output_notebook, show

output_notebook()

from bokeh.plotting import figure, output_file, show

from bokeh.models import ColumnDataSource

from bokeh.palettes import Category20 as palette



class EcosferaException(Exception):

    """Base class for other exceptions"""

    pass



class bcolors:

    HEADER = '\033[95m'

    BLUE = '\033[94m'

    GREEN = '\x1b[1m\x1b[32m'#'\033[92m'

    RED = '\x1b[1m\x1b[31m'#'\033[93m'

    FAIL = '\033[91m'

    ENDC = '\x1b[0m'#'\033[0m'

    BOLD = '\033[1m'

    UNDERLINE = '\033[4m'



os.system('color 4')



class WorldFactory:



    def __init__(self):

        self.entityTypes = ['Shrimp', 'Algae', 'Bacteria']

        self.lasteid = {k:0 for k in self.entityTypes}

        self.entitiesBase = []



    def printWorld(self, entities, pool):

        print(entities)

        entitiesStr = " ".join(cp.eid for e in entities for cp in e.cps)

        print(entitiesStr)

        print(pool)



    def createWorld(self, serializedWorld):

        entities = []

        pool = serializedWorld['pool']

        self.entitiesBase = serializedWorld['entitiesBase']



        entities = [self.createEntity(entityDict) for entityDict in serializedWorld['entities']]



        return entities, pool



    def createBaseEntity(self, entityType):

        # create base instance

        entityName = self.nextName(entityType)



        cps = [self.createConsumerProducer(entityName, entityType, c) for c in self.entitiesBase[entityType]['cycles'].keys()]

        

        return Entity(entityName, entityType, cps)



    def createEntity(self, serializedEntity):



        cps = []

        # create base instance

        entityType = serializedEntity['entityType']

        entity = self.createBaseEntity(entityType)

        

        for cp in entity.cps:

            # TODO update dictionary of serializedEntity instead of updated the accumulated value

            # update if it's there

            cp.accumulated = serializedEntity.get('cycles', {}).get(cp.cycle,{}).get('accumulated', cp.accumulated)



        return entity



    def nextName(self, entityType):

        entityName = entityType+str(self.lasteid[entityType])

        self.lasteid[entityType] += 1

        return entityName



    def unpackCPValues(self, entityType, cycleName):



        cycleDict = self.entitiesBase[entityType]['cycles'][cycleName]

        

        consumeDict = cycleDict['consume'] # 'consume': [('CO2', 1, 2)],

        produceDict = cycleDict['produce']



        arates      = cycleDict['rates']

        startAccumulated = cycleDict['accumulated']

        

        birthoffset = cycleDict['birthoffset']

        birthmin    = cycleDict['birthmin']



        return consumeDict, produceDict, arates, startAccumulated, birthoffset, birthmin



    def createConsumerProducer(self, entityName, entityType, cycleName):

        if entityType in self.entityTypes:

            neweid = entityName+'_'+cycleName



            consumeDict, produceDict, arates, startAccumulated, birthoffset, birthmin = self.unpackCPValues(entityType, cycleName)



            newCP = ConsumerProducer(

                entityType=entityType,

                entityName=entityName,

                eid=neweid,

                cycle=cycleName,

                consume=consumeDict,

                produce=produceDict,

                rates=arates,

                accumulated=startAccumulated,

                birthoffset=birthoffset,

                birthmin=birthmin

            )

            return newCP



        raise EcosferaException(f"Unknown ConsumerProducer Type '{entityType}'")





class Birther:



    def __init__(self, worldFactory, entities):

        self.worldFactory = worldFactory

        self.entities = entities



    def birthable(self, entity):

        return all(cp.birthable() for cp in entity.cps)



    def birth(self, parent, tickCount=None):

        entity = self.worldFactory.createBaseEntity(parent.entityType)

        parentStrBeforeBirth = parent.toShortString()

        for cp in entity.cps:

            cp.accumulated = cp.birthoffset

        for cp in parent.cps:

              cp.accumulated -= cp.birthoffset

        print(f"[{tickCount}]{bcolors.GREEN}{parentStrBeforeBirth} gives birth to {entity.toShortString()} with a cost of {[cp.birthoffset for cp in entity.cps]}{bcolors.ENDC}")

        self.entities.append(entity)

        return entity



    def tick(self, tickCount = None):

        return [self.birth(e, tickCount) for e in self.entities if self.birthable(e)]





class ConsumerProducer:



    def __init__(self, entityName, entityType, eid, cycle, consume, produce, rates, accumulated, birthoffset, birthmin):

        self.entityName = entityName

        self.entityType = entityType

        self.eid = eid

        self.reason = None

        self.accumulated = accumulated

        self.consumeRecipe = consume # {'O2': 1, 'CFood': 1}

        self.produceRecipe = produce # {'CO2': 1}

        self.rates = rates

        self.consumeTimer = 0

        self.produceTimer = 0

        self.birthoffset = birthoffset

        self.birthmin = birthmin

        self.cycle = cycle



    def toString(self):

        return f"{self.eid} {self.consumeRecipe} {self.produceRecipe} {self.rates} {self.accumulated} {self.reason}"



    def displayName(self):

        return self.eid



    def consume(self, pool):

        recipePoolResult = {k: (pool[k] - v) for k,v in self.consumeRecipe.items()}

        if any(v < 0 for v in recipePoolResult.values()):

            self.reason = {k:v for k, v in recipePoolResult.items() if v < 0}

            return False

        self.accumulated = self.accumulated+1

        pool.update(recipePoolResult)

        return True



    def produce(self, pool):

        if self.accumulated > 0:

            self.accumulated = self.accumulated-1

            newpool = pool

            newpool.update({k: pool[k] + v for k,v in self.produceRecipe.items()})

            return True

        else:

            return False



    def breakout(self, pool):

        pool.update({k: pool[k] + v*self.accumulated for k,v in self.produceRecipe.items()})

        self.accumulated = 0



    def snapshot(self):

        return f"{self.eid}: {self.accumulated}"



    def birthable(self):

        return self.accumulated > self.birthmin and self.accumulated > self.birthmin + self.birthoffset



    def tick(self, pool):

        consumed = None

        if self.consumeTimer <= 0:

            consumed = self.consume(pool)

            self.consumeTimer = self.rates[0]

        else:

            self.consumeTimer -= 1



        produced = None

        if self.produceTimer <= 0:

            produced = self.produce(pool)

            self.produceTimer = self.rates[1]

        else:

            self.produceTimer -= 1



        return consumed, produced





class Entity:



    def __init__(self, name, entityType, cps):

        self.name = name

        self.entityType = entityType

        self.cps = cps



    def toString(self):

        return " ".join(f"{self.name} {cp.toString()}" for cp in self.cps)



    def toShortString(self):

        return " ".join(f"{self.name} {cp.cycle} {cp.accumulated}" for cp in self.cps)



class Game:

    def __init__(self, birther, entities, pool):

        self.tickCount = 0

        self.entities = entities

        self.pool = pool

        self.poolhistory = []

        self.poolhistory.append(self.pool)

        self.log = []

        self.birther = birther

        self.plotdata = {k:(1, [v]) for (k,v) in pool.items()}

        self.plotdata.update({cp.eid: (1,[cp.accumulated]) for e in entities for cp in e.cps})



    # def loadWorld(self, yaml):

    #     shrimp1_C = ConsumerProducer(entityName="Shrimp1", eid="Shrimp1_C", consume={'O2': 1}, produce={'CO2': 1}, rates=(4,6), accumulated=10)

    #     self.cps.append(shrimp1_C)

    #     algae1_C = ConsumerProducer(entityName="Algae1", eid="Algae1_C", consume={'CO2': 1}, produce={'O2': 1}, rates=(2,3), accumulated=10)

    #     self.cps.append(algae1_C)

    #

    #     self.cps.sort(key=lambda cp: cp.eid)

    #     print(" ".join(f"{cp.eid}" for cp in self.cps))



    def addEntity(self, entity):

        self.entities.append(entity)



    def isDead(self):

        return len(self.entities) <= 0



    def setState(self, entities, pool):

        self.entities = entities

        self.pool = pool



    def savePlotPoint(self, pool, entities, tick, births, deadEntities=None):

        [self.plotdata[k][1].append(v) for k,v in pool.items()]

        [self.plotdata[cp.eid][1].append(cp.accumulated) for e in entities if e not in births for cp in e.cps]

        self.plotdata.update({cp.eid: (tick, [cp.accumulated]) for e in births for cp in e.cps})

        



    def tick(self):

        deadEntities = []



        for e in self.entities:

            dead = False

            for cp in e.cps:

                didConsume, didProduce = cp.tick(self.pool)

                if didConsume is False:

                    self.log.append(f"{self.tickCount} {cp.displayName()} failed consumption - {cp.reason}")

                if didProduce is False:

                    dead = True

                    self.log.append(f"{cp.eid} will die because {cp.reason}")

                    print(f"[{self.tickCount}] {bcolors.RED}{cp.eid} will die because {cp.reason}{bcolors.ENDC}")

            if dead:

                deadEntities.append(e)

                [cp.breakout(self.pool) for cp in e.cps]



        [self.entities.remove(e) for e in deadEntities]

        

        # births

        births = self.birther.tick(self.tickCount)

        

        self.poolhistory.append(self.pool)

        self.savePlotPoint(self.pool, self.entities, self.tickCount, births, deadEntities)

        self.tickCount += 1



    def snapshot(self):

        if not self.entities:

            return str(self.pool) + ' -- ' + "Dead Ecosystem"

        self.entities.sort(key=lambda e: e.name)

        entitiesSnapshot = (" ".join(f"{cp.eid}: {cp.accumulated}\t" for e in self.entities for cp in e.cps))

        return str(self.pool) + ' -- ' + entitiesSnapshot



    def toString(self):

        entitiesStr = "\n".join(e.toString() for e in self.entities)

        return str(self.pool) + '\n' + entitiesStr





def plotEcosystem(plotdata, result):

    

    print("Generating graph...")



    #on spyder, to show plots, run %matplotlib auto

    

    # output to static HTML file

    output_file("ecosfera_world1.html")



    # create a new plot with a title and axis labels

    p = figure(title=f"Ecosfera - {result}", tools="pan,box_zoom,wheel_zoom,zoom_in,zoom_out,reset,save",

               x_axis_label='time', y_axis_label='quantity', plot_width=1000, toolbar_location="above")



    # xs = [ [i for i in range(v[0],len(v[1]))] for k,v in g.plotdata.items()]

    # ys = [ v[1] for k,v in g.plotdata.items()]

    colors = palette[20]

    

    i = 0

    for k,v in plotdata.items():

        if i%10 == 0:

            print(f"Generating {i}/{len(plotdata.items())} entity")

        lastx = v[0] + len(v[1])

        x = [t for t in range(v[0],lastx)]

        y = v[1]

        p.line(x, y, legend_label=k, line_width=2, line_color=colors[i%20], alpha=0.8)

        if v[0] > 1:

            p.circle(v[0], v[1][0], line_color='green')

        p.circle(lastx, v[1][-1], line_color='red')

        i += 1

    

    print(f"{i}/{len(plotdata.items())}")

    

    # add a line renderer with legend and line thickness

    #p.multi_line(xs, ys)

    # p.multi_line(xs='xdata', ys='ydata', source=ColumnDataSource(data), line_color='color')



    

    p.legend.location = "top_left"

    p.legend.click_policy="hide"

    

    print("Graph generated")



    # show the results

    show(p)





def runWorld(world, nticks, verbose):

    

    factory = WorldFactory()



    entities, pool = factory.createWorld(world_selected)

    factory.printWorld(entities, pool)

    

    birther = Birther(factory, entities)

    

    g = Game(birther, entities, pool)

    print("\n")

    print(g.toString())

    print("\nStart simulation\n")

    for i in range(nticks):

        g.tick()

        if verbose > 0:

            print(f"{g.tickCount}: {g.snapshot()}")

        if g.isDead():

            break



    if not g.isDead():

        print(f"\n{bcolors.GREEN}Ecosystem survived {nticks} ticks{bcolors.ENDC}")

        result = "Survived"

    else:

        print(f"\n{bcolors.RED}Ecosystem is dead by {g.tickCount} ticks{bcolors.ENDC}")

        result = "Dead"



    plotEcosystem(g.plotdata, result)
world_full = {

    'pool':

    {

        'O2': 3,

        'CO2': 3,

        'CFood': 0,

        'NFood': 0,

        'Shit': 0,

        'NH3': 3,

    },

    'entities':

    [

        {

         'entityType': 'Shrimp',

        },

        {

         'entityType': 'Shrimp',

        },

        {

         'entityType': 'Shrimp',

        },

        {

         'entityType': 'Algae',

        },

        {

         'entityType': 'Algae',

        },

        {

         'entityType': 'Bacteria',

        },

        {

         'entityType': 'Bacteria',

        },

        {

         'entityType': 'Bacteria',

        }

    ],

    'entitiesBase':

    {

        'Shrimp':

        {

             'cycles':

             {

                 'C':

                 {

                     'consume': {'O2': 1, 'CFood': 1},

                     'produce': {'CO2': 1},

                     'rates': (40, 50),

                     'accumulated': 10,

                     'birthoffset': 10,

                     'birthmin': 15,

                 },

                 'N':

                 {

                     'consume': {'NFood': 1},

                     'produce': {'Shit': 1},

                     'rates': (50, 60),

                     'accumulated': 10,

                     'birthoffset': 10,

                     'birthmin': 15,

                 },

             }

        },

        'Algae':

        {

             'cycles':

             {

                 'C':

                 {

                     'consume': {'CO2': 1},

                     'produce': {'O2': 1, 'CFood': 1},

                     'rates': (10, 20),

                     'accumulated': 5,

                     'birthoffset': 10,

                     'birthmin': 20,

                 },

                 'N':

                 {

                     'consume': {'NH3': 1},

                     'produce': {'NFood': 1},

                     'rates': (30, 40),

                     'accumulated': 5,

                     'birthoffset': 10,

                     'birthmin': 20,

                 },

             }

        },

        'Bacteria':

        {

             'cycles':

             {

                 'N':

                 {

                     'consume': {'Shit': 1},

                     'produce': {'NH3': 1},

                     'rates': (50, 60),

                     'accumulated': 10,

                     'birthoffset': 20,

                     'birthmin': 40,

                 },

             }

        },

    }

}
world_selected = {

    'pool':

    {

        'O2': 30,

        'CO2': 30,

        'CFood': 0,

        'NFood': 0,

        'Shit': 0,

        'NH3': 30,

    },

    'entities':

    [

        {

         'entityType': 'Shrimp',

        },

        {

         'entityType': 'Shrimp',

        },

        {

         'entityType': 'Shrimp',

        },

        {

         'entityType': 'Algae',

        },

        {

         'entityType': 'Algae',

        },

        {

         'entityType': 'Bacteria',

        },

        {

         'entityType': 'Bacteria',

        },

        {

         'entityType': 'Bacteria',

        }

    ],

    'entitiesBase':

    {

        'Shrimp':

        {

             'cycles':

             {

                 'N':

                 {

                     'consume': {'NFood': 1},

                     'produce': {'Shit': 1},

                     'rates': (10, 50),

                     'accumulated': 10,

                     'birthoffset': 10,

                     'birthmin': 15,

                 },

             }

        },

        'Algae':

        {

             'cycles':

             {

                 'N':

                 {

                     'consume': {'NH3': 1},

                     'produce': {'NFood': 1},

                     'rates': (20, 30),

                     'accumulated': 10,

                     'birthoffset': 10,

                     'birthmin': 20,

                 },

             }

        },

        'Bacteria':

        {

             'cycles':

             {

                 'N':

                 {

                     'consume': {'Shit': 1},

                     'produce': {'NH3': 1},

                     'rates': (40, 60),

                     'accumulated': 10,

                     'birthoffset': 20,

                     'birthmin': 40,

                 },

             }

        },

    }

}



runWorld(world_selected, 20000, 0)