import pandas as pd
import geopandas as gpd
import cbsodata
import requests
# Zoek op welke data beschikbaar is
metadata = pd.DataFrame(cbsodata.get_meta('83765NED', 'DataProperties'))
metadata.head()

# Download geboortecijfers en verwijder spaties uit regiocodes
data = pd.DataFrame(cbsodata.get_data('83765NED', select = ['WijkenEnBuurten', 'Codering_3', 'GeboorteRelatief_25']))
data['Codering_3'] = data['Codering_3'].str.strip()
#De geodata wordt via de API van het Nationaal Georegister van PDOK gedownload en vervolgens ingelezen met read_file uit geopandas.

# Haal de kaart met gemeentegrenzen op van PDOK
geodata_url = 'https://geodata.nationaalgeoregister.nl/cbsgebiedsindelingen/wfs?request=GetFeature&service=WFS&version=2.0.0&typeName=cbs_gemeente_2017_gegeneraliseerd&outputFormat=json'
gemeentegrenzen = gpd.read_file(geodata_url)
#De geboortedata kan nu gekoppeld worden aan de gemeentegrenzen met merge.

# Koppel CBS-data aan geodata met regiocodes
gemeentegrenzen = pd.merge(gemeentegrenzen, data,
                           left_on = "statcode", 
                           right_on = "Codering_3")
#Tot slot kan de thematische kaart gemaakt worden met de functie plot.

# Maak een thematische kaart
p = gemeentegrenzen.plot(column='GeboorteRelatief_25', 
                         figsize = (10,8))
p.axis('off')
p.set_title('Levend geborenen per 1000 inwoners, 2017')