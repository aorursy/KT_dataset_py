#Start mit 4 Kannibale und 4 Missionäre in einem Boot.

start = [4,4,1]

ziel = [0,0,0]

def gib_zustand(zustand):

    

    #Ein leeres Arrayfeld um die Ergebnise abspeichern zu können.

    result =[]

    m,k,b = zustand

    new_b = 0 if b else 1

    

    if(b):

    #wenn b zutrifft guckt man ob Kannibalen größer als 0 sind.Wenn das zutrifft werden die Missionäre überprüft ob die größer als 0 sind.

        if (k>0):

            if(m>0):

                #Kannibalen und Missionäre werden subthrariert und in die leere Array hinzugefügt 

                result.append([m-1,k-1,new_b])

                #Missionäre größer als 1

                if(m>1):

                    #damit die missioäre nicht im unterzahl sind wird es -2 gemacht.

                    result.append([m-2,k-1,new_B])

                #Kannibal größer als 1

                if(k>1):

                    #Missiänre bleiben gleich, weil die Missionäre nicht im unterzahl sein dürfen

                    result.append([m,k-2,new_b])

                    #gleich regel wie bei k>1

                if(k>2):

                    result.append([m,k-3,new_b])

                    #alle 3 missinöre sind im boot

                if(m>2)    :

                    result.append([m-3,k,new_b])

                if(m>1):

                     result.append([m-2,k,new_B])   

                if(m>0):

                    result.append([m-1,k,new_b])

                    

        if not (b):

            if(k<4):

                if(m<4):

                    #Missionäre + 2 weil Sie im Überzahl sein müssen

                    result.append([m+2,k+1,new_b])

                    if(m<3):

                        result.append([m+1,k+1,new_b])

                    if(k<3):

                            result.append([m,k+2,new_b])

                    if(k<2):

                        result.append([m,k+2,new_b])

                        result.append([m,k+1,new_b]) 

                    if(m<2):

                        result.append([m+2,k,new_b])

                    if(m<3):

                        result.append([m+1,k,new_b])

        return result

    

    print(gib_zustand([3,3,1]))

    print(gib_zustand([4,4,1]))

                            

            



# Any results you write to the current directory are saved as output.