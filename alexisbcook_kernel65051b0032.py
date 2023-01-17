import folium

m_1 = folium.Map(location=[35,136], tiles='cartodbpositron', zoom_start=5)
def embed_map(m, file_name):
    from IPython.display import IFrame
    m.save(file_name)
    return IFrame(file_name, width='100%', height='500px')
embed_map(m_1, 'q_1.html')