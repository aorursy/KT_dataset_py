import random

import matplotlib.pyplot as pyp



class Unit_opration:

    def __init__( self , rpm ):

        self.processing_rate = rpm

        self.current_task = None

        self.time_remaining = 0    

    def start( self , task ):

        self.current_task = task

        self.time_remaining = task.get_wetness() * task.get_trash() // self.processing_rate

    def tick( self ):

        if self.current_task != None:

            self.time_remaining = self.time_remaining - 1

            if self.time_remaining <= 0:

                self.current_task = None    

    def busy( self ):

        if self.current_task != None:

            return True

        else:

            return False

    def get_t_remaining( self ):

        return self.time_remaining



class Package:

    def __init__( self , time ):

        self.timestamp = time

        self.wetness = random.randrange( 60 , 80 )

        self.trash = random.randrange( 6 , 12 )

    def get_stamp( self ):

        return self.timestamp

    def get_wetness( self ):

        return self.wetness

    def get_trash( self ):

        return self.trash

    def get_waiting_time( self , current_time ):

        return current_time - self.timestamp

    

class Queue():

    def __init__( self ):

        self.list = []

    def is_empty( self ):

        return self.list == []

    def enqueue( self , item ):

        self.list.insert( 0 , item )

    def dequeue( self ):

        return self.list.pop()

    def get_size( self ):

        return len( self.list )

    

def new_submission():

    num = random.randrange( 1 , 181 )

    if num >= 150:

        return True

    else:

        return False



def Processing_simulation( time_period_in_sec , ppm ):

    Processor = Unit_opration( ppm )

    Storage = Queue()

    waiting_times = []

    pack_counter = 0

    proc_ini_cnter = 0

    

    for sec in range( time_period_in_sec ):

        if new_submission():

            pack_counter += 1

            print( 'Second' , sec , ': Package' , pack_counter , 'arrived to the storage!' )

            package = Package( sec )

            Storage.enqueue( package )

        if ( not Processor.busy() ) and ( not Storage.is_empty() ):

            proc_ini_cnter += 1

            print( 'Second' , sec , ': Started to process package' , proc_ini_cnter )

            new_package = Storage.dequeue()

            Processor.start( new_package )

            waiting_times.append( new_package.get_waiting_time( sec ) )

        if Processor.get_t_remaining() - 1 == 0:

            print( 'Second' , sec , ': Done with package' , proc_ini_cnter )

        else:

            print("|")

        Processor.tick()



        

    average_waiting_time = sum( waiting_times ) / len( waiting_times )

    

    print( "\n\n\nAverage waiting time for a package to be processed is%6.2f secs.\nStill%3d packages are remaining." % ( average_waiting_time , Storage.get_size() ) )
Processing_simulation( 60 , 60 )