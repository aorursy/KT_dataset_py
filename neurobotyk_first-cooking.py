# Hi its me, the narrator i am just hiding here in the comments section, dont worry Iggy and Sophy or even Linda cant sem me here so i can explain to you what is happening in here.



# Linda once told sophie how to cut, and this is a DEFINITION of cutting that Sophie remembered

def slice_veggies(carrot):

    print(carrot.split(" "))



slice_veggies("one of carrots")
make_moms_favourite_cocoa()
# mom wanted sophie to slice carrots to even chunks 

def slice_veggies(carrot):

    half = int(len(carrot)/2)            # len means lenght

    carrot_part_1 = carrot[:half]        # [:] is scalled slicing operator

    carrot_part_2 = carrot[half:]        # second part is from half till end

    print(carrot_part_1,carrot_part_2)





slice_veggies("still one of carrots")

# than She turned to iggy to tell him what the defininion of good cocoa

def make_moms_favourite_cocoa():

    # but it was a secret ;)

    return "Delicious coffe"

print(make_moms_favourite_cocoa())
def slice_veggies(carrots):

    for carrot in carrots:

        slice_thickness=2

        tip_of_a_carrot = 0

        carrot_is_cuttable = tip_of_a_carrot<(len(carrot)-slice_thickness)

        while (carrot_is_cuttable):

            carrot_part = carrot[tip_of_a_carrot:tip_of_a_carrot+slice_thickness]  

            print(carrot_part)

            tip_of_a_carrot+=slice_thickness    #move the knife 

                                                #(after a cut is done the tip of a carrot is in a different place)

        print(carrot[tip_of_a_carrot:])         #the rest of carrot