from emissionspatial import emissionSpatialLayer

from emissionglobals import appGlobal

from emissionconfig import appConfig



objConfig = appConfig("/kaggle/input/ds4gconfiguration/config_kaggle.json")

outputPath = "/kaggle/working/output/"



aGlobal = appGlobal(objConfig, outputPath)



layerList = objConfig.getLayerList()



def calculateEmissionFactor():

    """

        This method get power plant x,y location as dataframe

    """

    for layer in layerList:

        print('processing layer ' + layer)

        spl = emissionSpatialLayer(aGlobal, layer)    

        dictSubRegions = spl.getSpatialLayerNPArr()



        print("calculate Marginal emission factor for power plants")

        if objConfig.getLayerType(layer) == "powerplant_subregion":

            me = marginalemissionfactor(aGlobal)

            me.calculateMarginalEmissions(layer)