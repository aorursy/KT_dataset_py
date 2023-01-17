def porcuantomevoy(practicas,parcial):

    n=4

    practicas.sort()

    sumapract=0

    for i in range(2,n+1):

        sumapract=practicas[i-1]+sumapract

    promprct=sumapract/3

    exfinal=(95-promprct*3-parcial*3)/3

    print("Te vas por: ",exfinal)
practicas=[10,8,16,12]

parcial=14

porcuantomevoy(practicas,parcial)