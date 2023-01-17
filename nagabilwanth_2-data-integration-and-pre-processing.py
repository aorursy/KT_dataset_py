from emissionconfig import appConfig



objConfig = appConfig("/kaggle/input/ds4gconfiguration/config_kaggle.json")

outputPath = "/kaggle/working/output/"



# Get config elements

config = objConfig.getConfig()
from emissionspatial import emissionSpatialLayer

from emissionglobals import appGlobal

from emissionconfig import appConfig



objConfig = appConfig("/kaggle/input/ds4gconfiguration/config_kaggle.json")

outputPath = "/kaggle/working/output/"



aGlobal = appGlobal(objConfig, outputPath)



def getCountySpatialNPArray():

    spl = emissionSpatialLayer(aGlobal, "County")

    dictSubRegions = spl.getSpatialLayerNPArr()

from emissionspatial import emissionSpatialLayer

from emissionglobals import appGlobal

from emissionconfig import appConfig



objConfig = appConfig("/kaggle/input/ds4gconfiguration/config_kaggle.json")

outputPath = "/kaggle/working/output/"



aGlobal = appGlobal(objConfig, outputPath)



layerList = objConfig.getLayerList()



def getGeneratorXYLocaiton():

    """

        This method get power plant x,y location as dataframe

    """

    for layer in layerList:

        print('processing layer ' + layer)

        spl = emissionSpatialLayer(aGlobal, layer)    

        dictSubRegions = spl.getSpatialLayerNPArr()



        for subRegion in dictSubRegions:

            # get power plant x,y location as data frame

            dfGeoLocation = spl.getGeneratorXYLocation(subRegion)



from emissiondataset import emissionTimeseries



def getDatasetTimeseres(self):

    """

        This method read input rasters names, analyze it and return the given timeseries

        as dataframe

    """

    try:

        dTimeseries = emissionTimeseries(self.objAppConfig, self.rasterType)

        timeseries_df = dTimeseries.getTimeseriesDataFrame()

        return timeseries_df

    except Exception as e:

        print(e)

        print("Failed at emissionanalyzer.py - getDatasetTimeseres")

        
from emissionspatial import emissionSpatialLayer

from emissionglobals import appGlobal

from emissionconfig import appConfig



objConfig = appConfig("/kaggle/input/ds4gconfiguration/config_kaggle.json")

outputPath = "/kaggle/working/output/"



aGlobal = appGlobal(objConfig, outputPath)



layerList = objConfig.getLayerList()



def separateweatherRaster():

    """

        This method get power plant x,y location as dataframe

    """

    for layer in layerList:

        print('processing layer ' + layer)

        spl = emissionSpatialLayer(aGlobal, layer)    

        dictSubRegions = spl.getSpatialLayerNPArr()



        for subRegion in dictSubRegions:

            # get power plant x,y location as data frame. This dataframe contain capacity of power plant

            dfGeoLocation = spl.getGeneratorXYLocation(subRegion)

            

            if not dfGeoLocation.empty:

                rpE = emissionFactorAnalyzer(aGlobal, "RasterEmission")

                rpE.generateEF_SubRegion(dictSubRegions[subRegion], dfGeoLocation, subRegion, layer)

                    

                rpW = weatherAnalyzer(aGlobal, "RasterWeather")

                rpW.getWeather_subRegion(dictSubRegions[subRegion], subRegion, layer)